"""E2E tests driving v2 streaming through the real langgraph-sdk client.

The point of these is fidelity: they use ``client.threads.stream`` exactly
as an application (or the Vue/React ``useStream``) would, so passing them
proves wire compatibility with the stock SDK, not just our own wire shape.

Skipped unless the server has ``FF_V2_EVENT_STREAMING=true`` (else 503).
Uses the ``stress_test`` graph (no LLM) so the run is hermetic.
"""

import json

import httpx
import pytest
from langgraph_sdk import get_client

from aegra_api.settings import settings
from tests.e2e._utils import elog


def _base_url() -> str:
    url = settings.app.SERVER_URL
    assert url is not None
    return url


async def _v2_enabled() -> bool:
    """True if the server has v2 streaming on (a thread command returns non-503)."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=10.0) as http:
        client = get_client(url=_base_url())
        thread = await client.threads.create()
        resp = await http.post(
            f"/threads/{thread['thread_id']}/commands",
            json={"id": 0, "method": "run.start", "params": {}},
        )
        return resp.status_code != 503


async def _ensure_assistant() -> str:
    client = get_client(url=_base_url())
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    return assistant["assistant_id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_thread_stream_run_start_and_events() -> None:
    """The stock SDK starts a run and receives v2 events over the thread stream."""
    if not await _v2_enabled():
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

    assistant_id = await _ensure_assistant()
    client = get_client(url=_base_url())

    methods: list[str] = []
    lifecycle_events: list[str] = []
    async with client.threads.stream(assistant_id=assistant_id) as ts:
        await ts.run.start(input={"messages": [{"role": "user", "content": json.dumps({"delay": 0.1, "steps": 1})}]})
        async for event in ts.events:
            method = event.get("method")
            methods.append(method)
            if method == "lifecycle":
                lifecycle_events.append(event["params"]["data"]["event"])
            if "completed" in lifecycle_events or "failed" in lifecycle_events:
                break

    elog("sdk thread stream methods", methods)
    assert "lifecycle" in methods, f"no lifecycle event received; got {methods}"
    assert "completed" in lifecycle_events, f"run did not complete; lifecycle={lifecycle_events}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_receives_values_events() -> None:
    """The SDK receives values-channel events carrying the run's state."""
    if not await _v2_enabled():
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

    assistant_id = await _ensure_assistant()
    client = get_client(url=_base_url())

    value_payloads: list[dict] = []
    async with client.threads.stream(assistant_id=assistant_id) as ts:
        await ts.run.start(input={"messages": [{"role": "user", "content": json.dumps({"delay": 0.1, "steps": 1})}]})
        async for event in ts.events:
            if event.get("method") == "values":
                value_payloads.append(event["params"]["data"])
            if event.get("method") == "lifecycle" and event["params"]["data"]["event"] in ("completed", "failed"):
                break

    elog("sdk values events", value_payloads)
    assert value_payloads, "no values events received"
    assert any("messages" in payload for payload in value_payloads)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_raw_wire_body_matches_sdk_contract() -> None:
    """The stream endpoint accepts the exact body the SDK sends: {channels} only."""
    if not await _v2_enabled():
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

    client = get_client(url=_base_url())
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # No run started and no run_id in the body — must open (200), not 4xx.
    async with (
        httpx.AsyncClient(base_url=_base_url(), timeout=15.0) as http,
        http.stream("POST", f"/threads/{thread_id}/stream/events", json={"channels": ["lifecycle"]}) as resp,
    ):
        assert resp.status_code == 200, f"SDK-shaped body rejected: {resp.status_code}"
        await resp.aclose()
