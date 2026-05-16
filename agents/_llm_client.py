"""Nemotron LLM client wrapper.

The single chokepoint between agents and the NVIDIA NIM endpoint. Every
agent that wants reasoning + structured output + tool-calling goes through
here.

Key behaviors:

  1. **`/think` mode by default** — adds the `/think` directive as a
     separate system message so Nemotron returns its reasoning trace in the
     `reasoning_content` channel.
  2. **Streaming** — calls `on_thinking(paragraph)` for each accumulated
     thinking paragraph as the stream arrives. Each call becomes one
     `EventType.THINKING` event in the agent's lifecycle, surfacing as a
     `💭 _…_` line in the dashboard chat feed.
  3. **Structured output** — `think_and_answer_structured` enforces a
     Pydantic schema on the answer. The schema's JSON schema is injected
     into the system prompt; the response is JSON-extracted + validated.
  4. **Tool-calling** — `think_and_answer_with_tools` runs a multi-turn
     loop: model decides which tools to call (with what args), we dispatch,
     feed results back, continue until the model emits no more tool_calls.
     Final answer is enforced against a Pydantic schema like (3).

When OpenClaw docs land, this wrapper either gets replaced by their LLM
plumbing or sits underneath theirs — either way, the agent code calling
`self.llm.think_and_answer_structured(...)` doesn't change.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, TypeVar

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from config import COORDINATOR_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL, WORKER_MODEL


T = TypeVar("T", bound=BaseModel)

# Default token budget. /think mode is verbose — easily 1500-3000 tokens
# of reasoning before the structured answer starts. 6k is generous headroom.
_DEFAULT_MAX_TOKENS = 6000


class LLMError(Exception):
    """Raised on unrecoverable LLM failure (API error, malformed structured output)."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class NemotronClient:
    """Thin OpenAI-SDK wrapper around the NVIDIA NIM endpoint.

    One instance per agent run is fine; the underlying `OpenAI` client is
    cheap to construct and connection-pools internally.
    """

    def __init__(self, model: str | None = None) -> None:
        if not NVIDIA_API_KEY:
            raise LLMError(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and fill "
                "in your key from build.nvidia.com."
            )
        self.model = model or WORKER_MODEL
        self._client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

    # ------------------------------------------------------------------
    # Free-form: reasoning + plain text answer
    # ------------------------------------------------------------------
    def think_and_answer(
        self,
        system: str,
        user: str,
        *,
        on_thinking: Callable[[str], None] | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        no_think: bool = False,
    ) -> str:
        """Stream a `/think`-mode response. Returns the assembled answer text.

        `on_thinking` is called once per accumulated thinking paragraph
        (boundary = blank line, `\\n\\n`). Returns when the stream ends.

        Pass `no_think=True` to switch to `/no_think` directive — model
        skips the reasoning channel and returns the answer directly.
        ~3-5× faster but no visible CoT.
        """
        return self._call_stream(
            system=system,
            user=user,
            on_thinking=on_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=None,
            no_think=no_think,
        )

    # ------------------------------------------------------------------
    # Structured: same as above + JSON schema enforcement + Pydantic validation
    # ------------------------------------------------------------------
    def think_and_answer_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        on_thinking: Callable[[str], None] | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.2,
        retries: int = 1,
        model: str | None = None,
        no_think: bool = False,
    ) -> T:
        """Like `think_and_answer`, but the answer must be a JSON object matching `schema`.

        Structured output defaults to `COORDINATOR_MODEL` (Nemotron-Super-49B)
        regardless of which model the client was constructed with — the smaller
        Nano-9B drops required fields ~half the time on non-trivial schemas.
        Free-form reasoning still uses the caller's chosen model (typically 9B
        for worker agents); structured output is the expensive step you only
        pay for at handoffs.

        Override via `model=...` if you want to force a specific one.

        Raises `LLMError` if all retries fail.
        """
        structured_model = model or COORDINATOR_MODEL

        json_schema = schema.model_json_schema()
        required_fields = json_schema.get("required", [])
        required_str = (
            "REQUIRED fields (must all be present): "
            + ", ".join(f"`{f}`" for f in required_fields)
            if required_fields else ""
        )
        augmented_system = (
            f"{system}\n\n"
            "## OUTPUT FORMAT\n"
            "Your final answer (after any reasoning) MUST be a single valid "
            "JSON object matching the schema below. Output ONLY the JSON "
            "object — no surrounding prose, no markdown code fences, no "
            "commentary. The very first character of your answer must be "
            "`{` and the very last must be `}`.\n\n"
            f"{required_str}\n\n"
            f"```json\n{json.dumps(json_schema, indent=2)}\n```"
        )
        user_tail = (
            "\n\nRespond with ONLY a JSON object matching the schema above. "
            "Include EVERY required field. Start with `{` and end with `}`."
        )

        # Try strict json_schema mode first (server-side enforcement).
        # Some Nemotron deployments don't support this; if so, fall back to
        # json_object + prompt-only enforcement on the second attempt.
        strict_schema = _build_strict_json_schema(schema, json_schema)
        response_formats: list[dict] = [
            {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": strict_schema,
                    "strict": True,
                },
            },
            {"type": "json_object"},
        ]

        last_error: str = ""
        last_text: str = ""

        for attempt in range(retries + 1):
            user_msg = user + user_tail
            if attempt > 0:
                user_msg += (
                    f"\n\nYour previous answer was invalid: {last_error}\n"
                    "Fix it. Include every required field with the exact field "
                    "names from the schema. JSON only, no prose."
                )

            for rf in response_formats:
                try:
                    text = self._call_stream(
                        model=structured_model,
                        system=augmented_system,
                        user=user_msg,
                        on_thinking=on_thinking if attempt == 0 else None,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format=rf,
                        no_think=no_think,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    # json_schema unsupported → fall through to json_object
                    if "json_schema" in str(rf.get("type", "")) and (
                        "json_schema" in str(exc).lower()
                        or "response_format" in str(exc).lower()
                        or "400" in str(exc)
                    ):
                        logger.debug(
                            "json_schema mode rejected by endpoint, falling back: {}",
                            exc,
                        )
                        on_thinking = None  # don't double-emit thinking
                        continue
                    raise
            else:
                # Both response_format attempts errored; surface the last one
                last_error = "API rejected both response_format modes"
                continue
            last_text = text

            raw_json = _extract_json_block(text)
            if not raw_json:
                last_error = "no JSON object found in answer"
                continue

            try:
                return schema.model_validate_json(raw_json)
            except ValidationError as exc:
                last_error = str(exc)
                continue

        logger.warning(
            "Structured output failed after {} attempt(s). Last error: {}. "
            "Last raw: {!r}",
            retries + 1, last_error, last_text[:500],
        )
        raise LLMError(
            f"LLM failed to return valid JSON for {schema.__name__} "
            f"after {retries + 1} attempt(s): {last_error}"
        )

    # ------------------------------------------------------------------
    # Tool-calling loop
    # ------------------------------------------------------------------
    def think_and_answer_with_tools(
        self,
        system: str,
        user: str,
        schema: type[T],
        tools: list[dict[str, Any]],
        tool_dispatch: dict[str, Callable[..., Any]],
        *,
        on_thinking: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict[str, Any], Any], None] | None = None,
        max_iterations: int = 8,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> T:
        """Run a multi-turn tool-use loop, then enforce the final answer schema.

        Phase 1 — tool loop:
          The model gets `tools=[...]` and decides which to call. Each call
          we dispatch via `tool_dispatch[name](**args)`, append the result to
          the conversation, and re-invoke. Loop ends when the model emits no
          more tool_calls. `on_tool_call(name, args, result)` fires once per
          dispatched call.

        Phase 2 — structured composition:
          Once the model has stopped requesting tools, we ask it once more
          (no tools, strict json_schema) to compose the final structured
          answer based on the full conversation history.

        Thinking content from every turn surfaces through `on_thinking` so
        the dashboard chat feed shows the agent reasoning between tool calls.

        Raises `LLMError` if the loop overruns `max_iterations` or the final
        answer fails schema validation.
        """
        tool_model = model or COORDINATOR_MODEL

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "/think"},
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for iteration in range(max_iterations):
            try:
                resp = self._client.chat.completions.create(
                    model=tool_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
            except Exception as exc:  # noqa: BLE001
                raise LLMError(
                    f"Tool-use API call failed (iter {iteration}): {exc}"
                ) from exc

            msg = resp.choices[0].message

            # Surface thinking from this turn
            msg_dict = msg.model_dump()
            reasoning = (
                msg_dict.get("reasoning_content")
                or msg_dict.get("reasoning")
                or ""
            )
            if reasoning and on_thinking is not None:
                for paragraph in [
                    p.strip() for p in reasoning.split("\n\n") if p.strip()
                ]:
                    on_thinking(paragraph)

            # Append assistant message to conversation. We strip None fields
            # so the next request doesn't carry empty optional keys.
            assistant_record: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_record["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_record)

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                # Model is done with tools. Move to phase 2: ask for the
                # structured answer based on everything it found.
                break

            # Dispatch every tool call and append results
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                fn = tool_dispatch.get(name)
                if fn is None:
                    result: Any = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": f"{type(exc).__name__}: {exc}"}

                if on_tool_call is not None:
                    try:
                        on_tool_call(name, args, result)
                    except Exception:  # noqa: BLE001
                        logger.exception("on_tool_call callback raised — ignoring")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str)[:4000],
                })
        else:
            # while-else: max_iterations exceeded
            raise LLMError(
                f"Tool-use loop exceeded {max_iterations} iterations without "
                "the model emitting a tool-free response."
            )

        # Phase 2: compose the final structured answer from the conversation.
        # We synthesize a fresh user message summarizing what was found and
        # request the structured output.
        conversation_summary = _conversation_summary(messages)
        return self.think_and_answer_structured(
            system=(
                "You compose a structured strategy spec based on prior "
                "research. The conversation above includes tool outputs and "
                "your own reasoning. Now produce the final JSON answer."
            ),
            user=(
                "Conversation so far:\n\n"
                f"{conversation_summary}\n\n"
                "Now compose the final structured StrategySpec."
            ),
            schema=schema,
            on_thinking=on_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            model=tool_model,
        )

    # ------------------------------------------------------------------
    # Shared stream loop — used by free-form + structured methods
    # ------------------------------------------------------------------
    def _call_stream(
        self,
        system: str,
        user: str,
        *,
        on_thinking: Callable[[str], None] | None,
        max_tokens: int,
        temperature: float,
        response_format: dict | None = None,
        model: str | None = None,
        no_think: bool = False,
    ) -> str:
        directive = "/no_think" if no_think else "/think"
        kwargs: dict = dict(
            model=model or self.model,
            messages=[
                {"role": "system", "content": directive},
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        if response_format is not None:
            # Nemotron-NIM may or may not honor this; passing it is safe.
            kwargs["response_format"] = response_format

        stream = self._client.chat.completions.create(**kwargs)

        thinking_buf = ""
        answer_parts: list[str] = []

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            delta_dict = delta.model_dump() if hasattr(delta, "model_dump") else {}

            reasoning = delta_dict.get("reasoning_content") or delta_dict.get("reasoning") or ""
            content = delta_dict.get("content") or ""

            if reasoning:
                thinking_buf += reasoning
                while "\n\n" in thinking_buf:
                    paragraph, _, thinking_buf = thinking_buf.partition("\n\n")
                    paragraph = paragraph.strip()
                    if paragraph and on_thinking is not None:
                        on_thinking(paragraph)
            if content:
                answer_parts.append(content)

        tail = thinking_buf.strip()
        if tail and on_thinking is not None:
            on_thinking(tail)

        return "".join(answer_parts).strip()


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)


def _conversation_summary(messages: list[dict[str, Any]]) -> str:
    """Render an LLM-friendly summary of a tool-use conversation.

    Skip the leading system+user pair (already known to phase 2) and emit
    a compact view of each subsequent assistant message + tool result.
    """
    lines: list[str] = []
    for m in messages[3:]:  # skip the two system msgs + initial user msg
        role = m.get("role", "?")
        if role == "assistant":
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"[assistant] {content[:400]}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                lines.append(
                    f"[assistant called] {fn.get('name')}({fn.get('arguments', '')[:200]})"
                )
        elif role == "tool":
            content = (m.get("content") or "").strip()
            lines.append(f"[tool result] {content[:600]}")
    return "\n\n".join(lines)


def _build_strict_json_schema(model: type[BaseModel], schema: dict) -> dict:
    """Adapt a Pydantic-emitted JSON schema for OpenAI strict json_schema mode.

    OpenAI strict mode requires:
      - top-level `additionalProperties: false`
      - every key in `properties` listed in `required`
      - no `$ref` cycles (we leave those alone; Pydantic v2's flat models work)
    """
    out = dict(schema)
    out["additionalProperties"] = False
    if "properties" in out:
        out["required"] = list(out["properties"].keys())
    # Recursively apply to nested object schemas (one level deep is enough for
    # our agent schemas; deeper would need a full walker).
    if "properties" in out:
        for k, v in out["properties"].items():
            if isinstance(v, dict) and v.get("type") == "object":
                v["additionalProperties"] = False
                if "properties" in v:
                    v["required"] = list(v["properties"].keys())
    # Same for $defs (Pydantic v2 puts nested models there).
    if "$defs" in out:
        for _, defn in out["$defs"].items():
            if isinstance(defn, dict) and defn.get("type") == "object":
                defn["additionalProperties"] = False
                if "properties" in defn:
                    defn["required"] = list(defn["properties"].keys())
    return out


def _extract_json_block(text: str) -> str | None:
    """Pull a JSON object out of an LLM response.

    Strategy:
      1. If wrapped in ```json ... ``` (or just ``` ... ```), use that.
      2. Otherwise, take the substring between the first `{` and the last `}`.
      3. Return None if neither finds anything plausible.
    """
    text = text.strip()
    if not text:
        return None

    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1].strip()

    return None
