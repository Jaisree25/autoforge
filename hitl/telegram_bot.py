"""Telegram bot for approving HITL gates from a phone.

`python-telegram-bot` v22 is async-first. We spin up an asyncio event loop on
a background thread and own its lifecycle. Send operations from any thread
go through `asyncio.run_coroutine_threadsafe()`.

Threading model
---------------
- One background thread runs the asyncio loop forever.
- The bot polls Telegram for updates (long-polling) on that loop.
- Approve/Reject callbacks resolve the request via `queue.resolve()`,
  which signals any in-process waiter immediately.
- Outbound messages (`send_approval_request`, `notify`) are scheduled on
  the bot's loop from whatever thread calls them.

The bot is optional. If `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` aren't set,
the coordinator service simply omits the Telegram channel — dashboard-only
HITL still works.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
)

from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)

from hitl.approval_queue import ApprovalQueue


_CALLBACK_PREFIX = "af"  # short to keep callback_data under Telegram's 64-byte cap


class TelegramApprovalBot:
    """Background-thread Telegram bot for HITL gates.

    Responder format: ``"telegram:<user_id>"`` so the audit trail records
    which Telegram user clicked Approve/Reject.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        queue: ApprovalQueue,
    ) -> None:
        self.token = token
        self.chat_id = chat_id
        self.queue = queue
        self._loop: asyncio.AbstractEventLoop | None = None
        self._app: Application | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the bot in a background thread. Returns once the bot is ready."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="telegram-bot", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=15.0):
            raise RuntimeError("Telegram bot failed to start within 15s")
        logger.info("Telegram bot started (chat_id={})", self.chat_id)

    def stop(self) -> None:
        """Stop the asyncio loop and join the thread."""
        if self._loop is None:
            return
        loop = self._loop

        async def _shutdown() -> None:
            assert self._app is not None
            if self._app.updater is not None:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        try:
            fut.result(timeout=10.0)
        except Exception:  # noqa: BLE001
            logger.exception("Telegram bot shutdown failed (ignoring)")
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        logger.info("Telegram bot stopped")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        app = (
            Application.builder()
            .token(self.token)
            .build()
        )
        app.add_handler(CallbackQueryHandler(self._on_callback))
        self._app = app

        async def _startup() -> None:
            await app.initialize()
            await app.start()
            assert app.updater is not None
            await app.updater.start_polling()
            self._ready.set()

        try:
            loop.run_until_complete(_startup())
            loop.run_forever()
        except Exception:  # noqa: BLE001
            logger.exception("Telegram bot loop crashed")
            self._ready.set()  # unblock start() if something broke
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Send side (callable from any thread)
    # ------------------------------------------------------------------
    def send_approval_request(self, request: ApprovalRequest) -> None:
        """Push an approval message with inline Approve / Reject buttons."""
        coro = self._send_approval_async(request)
        self._schedule(coro)

    def notify(
        self,
        run_id: str,
        message: str,
        payload: dict[str, Any] | None = None,  # noqa: ARG002 — reserved
    ) -> None:
        """Push a notification (no buttons)."""
        text = f"\U0001F4E2 [{run_id}] {message}"
        self._schedule(self._send_text_async(text))

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------
    async def _send_approval_async(self, request: ApprovalRequest) -> None:
        assert self._app is not None
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "✅ Approve",
                    callback_data=f"{_CALLBACK_PREFIX}:a:{request.request_id}",
                ),
                InlineKeyboardButton(
                    "❌ Reject",
                    callback_data=f"{_CALLBACK_PREFIX}:r:{request.request_id}",
                ),
            ]
        ])
        # Telegram message body cap is 4096; truncate payload preview.
        preview = json.dumps(request.payload, indent=2, default=str)
        if len(preview) > 2000:
            preview = preview[:2000] + "\n... [truncated]"
        text = (
            f"\U0001F514 *Approval needed*\n"
            f"_run_: `{request.run_id}`\n"
            f"_agent_: `{request.agent.value}`\n\n"
            f"*{_escape_md(request.title)}*\n"
            f"{_escape_md(request.description)}\n\n"
            f"```json\n{preview}\n```"
        )
        await self._app.bot.send_message(
            chat_id=self.chat_id,
            text=text,
            reply_markup=keyboard,
            parse_mode="Markdown",
        )

    async def _send_text_async(self, text: str) -> None:
        assert self._app is not None
        await self._app.bot.send_message(chat_id=self.chat_id, text=text)

    async def _on_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()

        try:
            prefix, action, request_id = query.data.split(":", 2)
        except ValueError:
            logger.warning("Telegram: malformed callback_data {!r}", query.data)
            return
        if prefix != _CALLBACK_PREFIX:
            return  # not for us

        decision = (
            ApprovalDecision.APPROVED if action == "a"
            else ApprovalDecision.REJECTED if action == "r"
            else None
        )
        if decision is None:
            logger.warning("Telegram: unknown action {!r}", action)
            return

        user_id = query.from_user.id if query.from_user else "unknown"
        responder = f"telegram:{user_id}"
        response = ApprovalResponse(
            request_id=request_id,
            decision=decision,
            responder=responder,
            comment=f"via Telegram by user {user_id}",
        )

        try:
            self.queue.resolve(response)
        except LookupError as exc:
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(f"⚠️ Already resolved: {exc}")
            return

        # Strip the buttons + append a status footer to the original message.
        original = query.message.text or ""
        footer = f"\n\n→ *{decision.value.upper()}* by {responder}"
        try:
            await query.edit_message_text(
                text=original + footer, parse_mode="Markdown"
            )
        except Exception:  # noqa: BLE001 — Telegram occasionally rejects edits
            logger.warning("Telegram: could not edit original message")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _schedule(self, coro) -> None:
        """Schedule a coroutine on the bot's loop from any thread."""
        if self._loop is None:
            logger.warning("Telegram bot not started; dropping coroutine")
            return
        asyncio.run_coroutine_threadsafe(coro, self._loop)


# Telegram Markdown v1 reserves * _ ` [ ; we escape sparingly so titles/descs
# render reasonably even if they contain underscores.
def _escape_md(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
            .replace("_", "\\_")
            .replace("*", "\\*")
            .replace("`", "\\`")
            .replace("[", "\\[")
    )
