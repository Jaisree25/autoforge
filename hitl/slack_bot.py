"""Slack bot for approving HITL gates from Slack.

Follows the same threading model as telegram_bot.py.
The bot is optional. If SLACK_BOT_TOKEN/SLACK_CHANNEL_ID aren't set,
the coordinator service simply omits the Slack channel.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from loguru import logger
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)
from hitl.approval_queue import ApprovalQueue


class SlackApprovalBot:
    """Slack bot for HITL gates.

    Routing:
      - Approval requests + Coordinator messages → `self.channel` (the main
        "coordinator" channel).
      - Per-agent notifications → `self.channel_map[agent_name]` if mapped,
        else `self.channel`. The bot polls only `self.channel` for
        CONFIRM/REJECT replies — keeps approval flow in one place.

    Responder format: ``"slack:<user_id>"`` so the audit trail records
    which Slack user clicked Approve/Reject.
    """

    def __init__(
        self,
        token: str,
        channel: str,
        queue: ApprovalQueue,
        channel_map: dict[str, str] | None = None,
    ) -> None:
        self.client = WebClient(token=token)
        self.channel = channel
        self.channel_map = channel_map or {}
        self.queue = queue
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start polling Slack in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="slack-bot", daemon=True
        )
        self._thread.start()
        logger.info("Slack bot started (channel={})", self.channel)

    def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        logger.info("Slack bot stopped")

    # ------------------------------------------------------------------
    # Send side
    # ------------------------------------------------------------------
    def send_approval_request(self, request: ApprovalRequest) -> None:
        """Push an approval message. User replies CONFIRM/REJECT in the
        channel (NOT as a thread reply — the bot polls channel history
        only, not thread replies)."""
        # Candidate-pick gates list architectures so the user can choose
        # by typing 1/2/3. CONFIRM defaults to candidate #1.
        is_candidate_pick = request.payload.get("kind") == "candidate_pick"
        candidates = request.payload.get("candidates") or []

        body_lines = [
            f":bell: *Approval needed*",
            f"*{request.title}*",
            request.description,
        ]
        if is_candidate_pick and candidates:
            body_lines.append("")
            body_lines.append("*Candidates:*")
            for i, c in enumerate(candidates, start=1):
                name = c.get("name", "?")
                family = c.get("family", "?")
                library = c.get("library", "?")
                rationale = (c.get("rationale") or "")[:120]
                body_lines.append(
                    f"  *{i}.* `{name}` ({family} / {library}) — {rationale}"
                )
            body_lines.append("")
            body_lines.append(
                "_Reply `1`, `2`, ... to pick a candidate, `CONFIRM` for #1, "
                "or `REJECT` to abort. New message in this channel — NOT a "
                "thread reply._"
            )
        else:
            body_lines.append("")
            body_lines.append(
                "_Reply `CONFIRM` or `REJECT` as a new message in this "
                "channel (NOT a thread reply)._"
            )

        text = "\n".join(body_lines)
        try:
            self.client.chat_postMessage(
                channel=self.channel,
                text=text,
                blocks=[{"type": "section",
                         "text": {"type": "mrkdwn", "text": text}}],
            )
        except SlackApiError as e:
            logger.error("Slack send_approval_request failed: {}", e)

    def notify(
        self,
        run_id: str,
        message: str,
        payload: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        """Push a plain notification message.

        If `agent` is set and that agent has a mapped channel, post there.
        Otherwise post to the main channel. Run ID is added as a `[<id8>]`
        prefix so multiple concurrent runs are distinguishable in the feed.
        """
        target = self.channel_map.get(agent, self.channel) if agent else self.channel
        try:
            self.client.chat_postMessage(
                channel=target,
                text=f":mega: [{run_id[:8]}] {message}",
            )
        except SlackApiError as e:
            logger.error("Slack notify failed (channel={}): {}", target, e)

    # ------------------------------------------------------------------
    # Poll loop — checks for CONFIRM/REJECT messages
    # ------------------------------------------------------------------
    def _poll_loop(self) -> None:
        # Start the cursor 60 seconds in the past so we don't miss the
        # bot's own startup notification or any message that landed
        # between startup and the first poll. Slack ts precision and
        # Python time.time() precision can drift by microseconds, which
        # causes boundary messages to be filtered out — the 60s buffer
        # makes the filter robust.
        last_checked = f"{time.time() - 60:.6f}"
        poll_count = 0
        while self._running:
            try:
                result = self.client.conversations_history(
                    channel=self.channel,
                    oldest=last_checked,
                    limit=20,
                )
                messages = result.get("messages", [])
                poll_count += 1
                if messages:
                    logger.info(
                        "Slack poll #{}: {} message(s) since {} — "
                        "first text: {!r}",
                        poll_count, len(messages), last_checked,
                        (messages[0].get("text") or "")[:80],
                    )
                elif poll_count % 20 == 0:
                    # Heartbeat every ~60s so we know the bot is alive.
                    logger.info(
                        "Slack poll #{}: no messages since {} (heartbeat)",
                        poll_count, last_checked,
                    )
                for msg in reversed(messages):
                    text = msg.get("text", "").strip().upper()
                    user_id = msg.get("user", "unknown")
                    ts = msg.get("ts", "")

                    # Parse the user's reply: CONFIRM, REJECT, or a digit /
                    # "CHOOSE N" / "PICK N" for candidate-pick gates.
                    decision: ApprovalDecision | None = None
                    selected_index: int | None = None
                    if text in ("CONFIRM", "APPROVE", "YES", "Y"):
                        decision = ApprovalDecision.APPROVED
                    elif text in ("REJECT", "DENY", "NO", "N"):
                        decision = ApprovalDecision.REJECTED
                    else:
                        # Match a bare digit or "CHOOSE N" / "PICK N"
                        import re
                        m = re.match(
                            r"^(?:CHOOSE\s+|PICK\s+|#)?(\d+)$",
                            text,
                        )
                        if m:
                            decision = ApprovalDecision.APPROVED
                            # User-facing 1-indexed; coordinator uses 0-indexed.
                            selected_index = max(0, int(m.group(1)) - 1)

                    if decision is not None:
                        responder = f"slack:{user_id}"
                        pending_list = self.queue.store.list_pending_approvals(
                            run_id=None,
                        )
                        if pending_list:
                            pending = pending_list[-1]  # newest (list is ASC)
                            payload = (
                                {"selected_index": selected_index}
                                if selected_index is not None
                                else None
                            )
                            response = ApprovalResponse(
                                request_id=pending.request_id,
                                decision=decision,
                                response_payload=payload,
                                responder=responder,
                                comment=(
                                    f"via Slack by {user_id}"
                                    + (
                                        f" — picked #{selected_index + 1}"
                                        if selected_index is not None else ""
                                    )
                                ),
                            )
                            try:
                                self.queue.resolve(response)
                                self.client.chat_postMessage(
                                    channel=self.channel,
                                    text=(
                                        f":white_check_mark: "
                                        f"{decision.value.upper()} by "
                                        f"{responder} for "
                                        f"`{pending.title}`"
                                    ),
                                )
                                logger.info(
                                    "Slack {} by {} resolved approval {}",
                                    decision.value, responder,
                                    pending.request_id,
                                )
                            except LookupError as exc:
                                logger.warning(
                                    "Slack approval already resolved: {}", exc,
                                )
                        else:
                            logger.warning(
                                "Slack {} from {} but no pending approval",
                                text, user_id,
                            )

                if messages:
                    last_checked = messages[0].get("ts", last_checked)

            except SlackApiError as e:
                logger.error("Slack poll failed: {}", e)

            time.sleep(3)
