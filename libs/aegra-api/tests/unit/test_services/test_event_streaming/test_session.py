"""Tests for ThreadEventSession: thread-scoping, seq, channel filter, since."""

from collections.abc import Iterator

import pytest
from langchain_core.messages import AIMessageChunk

from aegra_api.services.broker import BrokerManager
from aegra_api.services.event_streaming import session as session_module
from aegra_api.services.event_streaming.session import (
    ThreadEventSession,
    validate_channels,
)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[BrokerManager]:
    """Swap in a fresh in-memory broker manager for the session under test."""
    mgr = BrokerManager()
    monkeypatch.setattr(session_module, "broker_manager", mgr)
    yield mgr


def _lister(*run_ids: str):
    """A run lister returning a fixed set of run ids."""

    async def list_run_ids() -> list[str]:
        return list(run_ids)

    return list_run_ids


async def _seed(mgr: BrokerManager, run_id: str, events: list[tuple[str, object]]) -> None:
    broker = mgr.get_or_create_broker(run_id)
    for i, raw in enumerate(events, start=1):
        await broker.put(f"{run_id}_event_{i}", raw)


def _chunk(text: str, *, msg_id: str = "m1", last: bool = False) -> AIMessageChunk:
    chunk = AIMessageChunk(content=text, id=msg_id)
    if last:
        chunk.chunk_position = "last"
    return chunk


def _make_session(
    thread_id: str, *, channels: set[str], run_ids: tuple[str, ...], since: int | None = None
) -> ThreadEventSession:
    return ThreadEventSession(
        thread_id,
        channels=channels,
        list_run_ids=_lister(*run_ids),
        since=since,
        idle_grace_seconds=0.0,
    )


async def _collect(session: ThreadEventSession) -> list[dict]:
    return [evt async for evt in session.stream()]


class TestThreadStreaming:
    async def test_message_stream_projects_protocol_events(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("messages", (_chunk("hello"), {})),
                ("messages", (_chunk(" world", last=True), {})),
                ("end", {"status": "success"}),
            ],
        )
        session = _make_session("t1", channels={"messages", "lifecycle"}, run_ids=("run-1",))
        events = await _collect(session)

        methods = [(e["method"], e["params"]["data"].get("event")) for e in events]
        assert methods == [
            ("messages", "message-start"),
            ("messages", "content-block-delta"),
            ("messages", "content-block-delta"),
            ("messages", "message-finish"),
            ("lifecycle", "completed"),
        ]

    async def test_envelope_wraps_payload_in_params_data(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", {"a": 1}), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",)))
        assert events[0]["params"] == {"data": {"a": 1}, "namespace": []}

    async def test_seq_is_monotonic_from_one(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", {"a": 1}), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",)))
        assert [e["seq"] for e in events] == [1, 2]

    async def test_seq_spans_multiple_runs_on_thread(self, manager: BrokerManager) -> None:
        """A thread's seq is continuous across the runs that execute on it."""
        await _seed(manager, "run-1", [("values", {"a": 1}), ("end", {"status": "success"})])
        await _seed(manager, "run-2", [("values", {"a": 2}), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1", "run-2")))
        # run-1: values(1) end(2); run-2: values(3) end(4)
        assert [e["seq"] for e in events] == [1, 2, 3, 4]

    async def test_channel_filter_drops_unsubscribed(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("values", {"a": 1}), ("updates", {"n": {"b": 2}}), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",)))
        assert {e["method"] for e in events} == {"values"}

    async def test_seq_is_absolute_not_filter_relative(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("values", {"a": 1}), ("updates", {"n": {"b": 2}}), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert [(e["method"], e["seq"]) for e in events] == [("lifecycle", 3)]

    async def test_since_skips_already_seen(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("values", {"a": 1}), ("values", {"a": 2}), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",), since=1))
        assert [e["seq"] for e in events] == [2, 3]

    async def test_lifecycle_interrupted(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "interrupted"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"event": "interrupted"}

    async def test_lifecycle_failed_carries_error(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("error", {"status": "error", "message": "boom"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"event": "failed", "error": "boom"}

    async def test_applied_through_seq_tracks_max(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", {"a": 1}), ("end", {"status": "success"})])
        session = _make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",))
        await _collect(session)
        assert session.applied_through_seq == 2

    async def test_namespaced_custom_subscription_receives_custom_events(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("custom", {"hello": "world"}), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"custom:my_event", "lifecycle"}, run_ids=("run-1",)))
        assert any(e["method"] == "custom" for e in events)

    async def test_empty_thread_closes_after_idle(self, manager: BrokerManager) -> None:
        """A thread with no runs ends the stream after the idle grace (0s here)."""
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=()))
        assert events == []


class TestValidateChannels:
    def test_valid_channels(self) -> None:
        valid, invalid = validate_channels(["messages", "values", "custom:foo"])
        assert valid == {"messages", "values", "custom:foo"}
        assert invalid == []

    def test_invalid_channels_collected(self) -> None:
        valid, invalid = validate_channels(["messages", "bogus"])
        assert valid == {"messages"}
        assert invalid == ["bogus"]

    def test_empty_list_is_error(self) -> None:
        valid, invalid = validate_channels([])
        assert valid == set()
        assert invalid

    def test_non_list_is_error(self) -> None:
        valid, invalid = validate_channels("messages")
        assert invalid
