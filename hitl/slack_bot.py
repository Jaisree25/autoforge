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
        """Push an approval message with Approve/Reject buttons."""
        try:
            self.client.chat_postMessage(
                channel=self.channel,
                text=f"*Approval needed*\n*{request.title}*\n{request.description}",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f":bell: *Approval needed*\n"
                                f"*{request.title}*\n"
                                f"{request.description}\n\n"
                                f"Reply `CONFIRM` to approve or `REJECT` to stop."
                            ),
                        },
                    }
                ],
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
        last_checked = str(time.time())
        while self._running:
            try:
                result = self.client.conversations_history(
                    channel=self.channel,
                    oldest=last_checked,
                    limit=10,
                )
                messages = result.get("messages", [])
                for msg in reversed(messages):
                    text = msg.get("text", "").strip().upper()
                    user_id = msg.get("user", "unknown")
                    ts = msg.get("ts", "")

                    if text in ("CONFIRM", "REJECT"):
                        decision = (
                            ApprovalDecision.APPROVED
                            if text == "CONFIRM"
                            else ApprovalDecision.REJECTED
                        )
                        responder = f"slack:{user_id}"

                        # Find the oldest pending request and resolve it
                        pending = self.queue.get_pending()
                        if pending:
                            response = ApprovalResponse(
                                request_id=pending.request_id,
                                decision=decision,
                                responder=responder,
                                comment=f"via Slack by {user_id}",
                            )
                            try:
                                self.queue.resolve(response)
                                self.client.chat_postMessage(
                                    channel=self.channel,
                                    text=f":white_check_mark: {decision.value.upper()} by {responder}",
                                )
                            except LookupError as exc:
                                logger.warning("Already resolved: {}", exc)

                if messages:
                    last_checked = messages[0].get("ts", last_checked)

            except SlackApiError as e:
                logger.error("Slack poll failed: {}", e)

            time.sleep(3)
