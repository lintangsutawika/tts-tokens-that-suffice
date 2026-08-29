"""
Node-side bridge for the Modal reverse relay (harbor/relay/modal_relay.py).

Runs on the vLLM node (same Slurm job as the server). Dials OUT to the relay's
WebSocket and holds it open; for each request frame the relay forwards, it calls
the local vLLM and sends the response back. This is the leg that makes a private
node reachable from Modal without accepting any inbound connection.

Env:
    RELAY_WS_URL    wss://<workspace>--tts-vllm-relay-web.modal.run/bridge
    RELAY_SECRET    shared secret (matches the Modal 'tts-relay' secret)
    VLLM_LOCAL_URL  local vLLM base, e.g. http://localhost:8000  (no trailing /v1)

Run (deps resolved by uv):
    uv run --with websockets --with httpx python harbor/relay/bridge.py
"""

import asyncio
import json
import os
import sys

import httpx
import websockets

RELAY_WS_URL = os.environ["RELAY_WS_URL"]
RELAY_SECRET = os.environ["RELAY_SECRET"]
VLLM_LOCAL_URL = os.environ.get("VLLM_LOCAL_URL", "http://localhost:8000").rstrip("/")


def log(msg: str) -> None:
    print(f"[bridge] {msg}", flush=True, file=sys.stderr)


async def handle(ws, send_lock: asyncio.Lock, frame: dict, client: httpx.AsyncClient) -> None:
    """Proxy one relayed request to the local vLLM and return the response frame."""
    try:
        url = VLLM_LOCAL_URL + frame["path"]
        if frame.get("query"):
            url += "?" + frame["query"]
        r = await client.request(
            frame["method"],
            url,
            content=frame["body"].encode(),
            headers={"content-type": "application/json"},
        )
        resp = {
            "id": frame["id"],
            "status": r.status_code,
            "body": r.text,
            "content_type": r.headers.get("content-type", "application/json"),
        }
    except Exception as e:  # return a 502 rather than dropping the request
        resp = {
            "id": frame["id"],
            "status": 502,
            "body": json.dumps({"error": {"message": f"bridge->vllm failed: {e}"}}),
            "content_type": "application/json",
        }
    # Serialize the vLLM response body (a completion, up to ~100s of KB) off the
    # event loop -- a synchronous json.dumps of a large payload here would block the
    # loop and starve the WebSocket keepalive (missed pongs -> the relay drops us
    # with a 1011). See main()'s ping settings.
    payload = await asyncio.to_thread(json.dumps, resp)
    # Serialize sends: concurrent handle() tasks writing the same socket would
    # interleave frames and trip a 1002 protocol error on the relay. If the socket
    # dropped while this request was in flight, log and drop -- the relay will time
    # the request out; a raw exception here is an unretrieved-task traceback.
    try:
        async with send_lock:
            await ws.send(payload)
    except Exception as e:
        log(f"dropping response {frame['id']}: ws send failed ({e})")


async def main() -> None:
    headers = {"Authorization": f"Bearer {RELAY_SECRET}"}
    async with httpx.AsyncClient(timeout=600.0) as client:
        while True:
            try:
                async with websockets.connect(
                    RELAY_WS_URL,
                    additional_headers=headers,
                    max_size=None,
                    # A momentarily-busy loop on either end must not be a death
                    # sentence: at 64-way concurrency the relay/bridge loops spend
                    # bursts shuttling large SWE-bench payloads, and a 20s pong
                    # deadline was tripping 1011 keepalive timeouts that dropped
                    # every in-flight request. Ping less often, wait far longer.
                    ping_interval=30,
                    ping_timeout=90,
                ) as ws:
                    log(f"connected to relay; forwarding to {VLLM_LOCAL_URL}")
                    send_lock = asyncio.Lock()  # one per connection
                    async for msg in ws:
                        # Parse the (large) request frame off the loop so keepalives
                        # keep flowing while a hefty prompt body deserializes.
                        frame = await asyncio.to_thread(json.loads, msg)
                        asyncio.create_task(handle(ws, send_lock, frame, client))
            except Exception as e:
                log(f"disconnected ({e}); reconnecting in 5s")
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
