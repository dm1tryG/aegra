"""Thread-scoped session that turns a thread's run events into v2 events.

A v2 stream is scoped to a *thread*, not a run (matching the LangGraph
SDK): the client opens the stream, then issues ``run.start``, and the
events of whatever run(s) execute on that thread flow through. So the
session discovers the thread's runs, tails each one's broker (the same
broker the legacy SSE path uses), and projects their raw events into
protocol channel events under one thread-monotonic ``seq``.

``seq`` is the reconnect cursor: a client resends the last ``seq`` it saw
as ``since`` and the session skips anything at or below it. ``event_id``
is a distinct unique-per-event string used by the client for dedup.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import structlog

from aegra_api.services.broker import broker_manager
from aegra_api.services.event_streaming.channels import is_supported_channel
from aegra_api.services.event_streaming.protocol import build_event
from aegra_api.services.event_streaming.translator import EventTranslator

logger = structlog.getLogger(__name__)

__all__ = ["RunLister", "ThreadEventSession", "validate_channels"]

# Map a run's terminal broker status to a lifecycle AgentStatus.
_STATUS_TO_LIFECYCLE: dict[str, str] = {
    "success": "completed",
    "completed": "completed",
    "interrupted": "interrupted",
    "error": "failed",
}

# How long to keep polling for a new run after the thread goes quiet before
# closing the stream. Covers the SDK gap between opening the stream and the
# run.start landing.
_IDLE_GRACE_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.25

# Async callable returning the thread's run ids, oldest first.
RunLister = Callable[[], Awaitable[list[str]]]


class ThreadEventSession:
    """Streams v2 events for all runs on a thread, filtered to channels."""

    def __init__(
        self,
        thread_id: str,
        *,
        channels: set[str],
        list_run_ids: RunLister,
        since: int | None = None,
        idle_grace_seconds: float = _IDLE_GRACE_SECONDS,
    ) -> None:
        self._thread_id = thread_id
        self._channels = channels
        self._list_run_ids = list_run_ids
        self._since = since
        self._idle_grace = idle_grace_seconds
        self._seq = 0
        self._translator = EventTranslator()
        self._drained: set[str] = set()

    @property
    def applied_through_seq(self) -> int:
        """The highest seq assigned so far (the SDK's initial cursor)."""
        return self._seq

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield v2 envelopes for the thread's runs until they finish + idle.

        Before any run has been drained we wait up to ``idle_grace`` for a
        ``run.start`` to land (the SDK opens the stream first). Once a run has
        completed we close promptly — a thread rarely starts a second run on
        the same open stream, and the client reconnects if it does.
        """
        idle_deadline: float | None = None
        drained_any = False
        loop = asyncio.get_running_loop()

        while True:
            progressed = False
            for run_id in await self._fresh_run_ids():
                async for envelope in self._drain_run(run_id):
                    progressed = True
                    yield envelope
                self._drained.add(run_id)
                drained_any = True

            if progressed:
                idle_deadline = None
                continue

            if drained_any:
                return

            # No run yet. Wait briefly for a run.start to land, then re-check;
            # give up after the idle grace so an empty thread doesn't hang.
            now = loop.time()
            if idle_deadline is None:
                idle_deadline = now + self._idle_grace
            elif now >= idle_deadline:
                return
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _fresh_run_ids(self) -> list[str]:
        """Run ids on the thread we haven't drained yet, oldest first."""
        return [run_id for run_id in await self._list_run_ids() if run_id not in self._drained]

    async def _drain_run(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Replay then tail one run's broker, projecting to v2 envelopes."""
        broker = broker_manager.get_or_create_broker(run_id)
        seen: set[str] = set()

        for event_id, raw_event in await broker.replay(None):
            seen.add(event_id)
            for envelope in self._project(event_id, raw_event):
                yield envelope
            if _is_terminal(raw_event):
                return

        async for event_id, raw_event in broker.aiter():
            if event_id in seen:
                continue
            for envelope in self._project(event_id, raw_event):
                yield envelope
            if _is_terminal(raw_event):
                return

    def _project(self, event_id: str, raw_event: Any) -> list[dict[str, Any]]:
        """Translate one raw broker event into filtered, seq'd envelopes."""
        mode, payload = _unwrap(raw_event)
        if mode is None:
            return []

        channel_events = (
            self._lifecycle(payload) if mode in ("end", "error") else self._translator.translate(mode, payload)
        )

        envelopes: list[dict[str, Any]] = []
        for channel, data, namespace in channel_events:
            # seq counts every translated event before channel filtering, so
            # the cursor is absolute position in the thread's stream — a
            # reconnect with a different channel set still resumes correctly.
            self._seq += 1
            if not self._wants(channel):
                continue
            if self._since is not None and self._seq <= self._since:
                continue
            envelopes.append(
                build_event(channel, data, namespace=namespace, seq=self._seq, event_id=f"{event_id}:{self._seq}")
            )
        return envelopes

    def _wants(self, channel: str) -> bool:
        """True if the client subscribed to this channel.

        The translator emits the base ``custom`` channel, so any custom
        subscription — plain ``custom`` or namespaced ``custom:<name>`` —
        matches it. Named filtering is a follow-up.
        """
        if channel == "custom":
            return any(c == "custom" or c.startswith("custom:") for c in self._channels)
        return channel in self._channels

    def _lifecycle(self, payload: Any) -> list[tuple[str, dict[str, Any], list[str]]]:
        """Build a lifecycle event from a terminal broker payload."""
        status = payload.get("status") if isinstance(payload, dict) else None
        event = _STATUS_TO_LIFECYCLE.get(status or "", "completed")
        data: dict[str, Any] = {"event": event}
        if isinstance(payload, dict) and (message := payload.get("message")):
            data["error"] = message
        return [("lifecycle", data, [])]


def _is_terminal(raw_event: Any) -> bool:
    """True for a run's final ``end`` / ``error`` broker event."""
    mode, _ = _unwrap(raw_event)
    return mode in ("end", "error")


def _unwrap(raw_event: Any) -> tuple[str | None, Any]:
    """Pull ``(mode, payload)`` out of a broker event; ``(None, None)`` if unknown."""
    if isinstance(raw_event, (tuple, list)) and len(raw_event) == 2:
        return raw_event[0], raw_event[1]
    return None, None


def validate_channels(channels: Any) -> tuple[set[str], list[str]]:
    """Split a requested channel list into (valid set, invalid names).

    Used by the route to reject unknown channels up front rather than
    opening a stream that never emits.
    """
    if not isinstance(channels, list) or not channels:
        return set(), ["channels must be a non-empty array"]
    valid: set[str] = set()
    invalid: list[str] = []
    for channel in channels:
        if isinstance(channel, str) and is_supported_channel(channel):
            valid.add(channel)
        else:
            invalid.append(str(channel))
    return valid, invalid
