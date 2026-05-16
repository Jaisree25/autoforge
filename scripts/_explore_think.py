"""One-off: see what Nemotron's `/think` mode actually returns so we can
write the parser correctly. Discard after we lock in the format.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI

from config import NVIDIA_API_KEY, NVIDIA_BASE_URL, WORKER_MODEL


def probe_non_streaming() -> None:
    print("=" * 80)
    print("non-streaming  /think  WORKER_MODEL")
    print("=" * 80)
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    resp = client.chat.completions.create(
        model=WORKER_MODEL,
        messages=[
            {"role": "system", "content": "/think"},
            {"role": "user", "content":
                "What is 5*7? Think briefly, then give just the number. "
                "Keep the whole response under 100 tokens."},
        ],
        max_tokens=500,
        temperature=0.1,
    )
    msg = resp.choices[0].message
    print("--- message dict (all fields) ---")
    print(msg.model_dump())
    print("--- content repr ---")
    print(repr(msg.content))
    print("--- usage ---")
    print(resp.usage.model_dump() if resp.usage else None)


def probe_streaming() -> None:
    print("=" * 80)
    print("streaming  /think  WORKER_MODEL")
    print("=" * 80)
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    stream = client.chat.completions.create(
        model=WORKER_MODEL,
        messages=[
            {"role": "system", "content": "/think"},
            {"role": "user", "content":
                "What is 5*7? Think briefly, then give just the number."},
        ],
        max_tokens=500,
        temperature=0.1,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        # Print every field on the first non-empty delta so we see all keys
        if chunks == [] or chunks[-1] != delta.model_dump():
            chunks.append(delta.model_dump())
    print(f"Total {len(chunks)} delta chunks. First 3 + last 2:")
    for c in chunks[:3]:
        print("  delta:", c)
    print("  ...")
    for c in chunks[-2:]:
        print("  delta:", c)

    full = "".join(c.get("content", "") or "" for c in chunks)
    print("--- assembled content (first 800 chars) ---")
    print(full[:800])


if __name__ == "__main__":
    if not NVIDIA_API_KEY:
        print("NVIDIA_API_KEY missing"); sys.exit(1)
    probe_non_streaming()
    print()
    probe_streaming()
