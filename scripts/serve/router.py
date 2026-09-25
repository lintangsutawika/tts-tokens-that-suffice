"""Prefix-affinity router for a fleet of independent vLLM servers.

vLLM's built-in DP load balancer cannot handle bursty arrivals: its engine-load
table is a 100ms-stale snapshot from the coordinator that wipes the local
in-flight increments on every refresh, and ties break toward the scan start
index. Measured on 8x MI250X with 64 mini-swe workers, that left 5-6 of 8 GCDs
at *literally zero* requests for 7 minutes straight while the rest queued and
preempted. Neither --api-server-count 8 (3 engines used) nor 1 (2 engines used)
is acceptable.

This router replaces it with the two things the internal LB structurally lacks:

  1. Exact in-flight counts. We know precisely how many requests are on each
     backend because we are the only one dispatching to them.
  2. Conversation affinity. A SWE-agent resends its entire history every step,
     so pinning a conversation to one backend turns that history into a prefix
     cache hit instead of a full reprefill.

A conversation is identified by its first two messages (system + task), which
are stable across every step of a mini-swe trajectory. First time we see one, it
goes to the least-loaded backend; after that it stays there.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

BACKENDS = [b for b in os.environ.get("BACKENDS", "").split(",") if b]
PORT = int(os.environ.get("ROUTER_PORT", "9142"))
# Long: a SWE step can legitimately take minutes when the backend is busy.
TIMEOUT = httpx.Timeout(connect=10.0, read=3600.0, write=60.0, pool=60.0)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [router] %(message)s", stream=sys.stdout
)
log = logging.getLogger("router")

app = FastAPI()

inflight: list[int] = [0] * len(BACKENDS)
affinity: dict[str, int] = {}
lock = asyncio.Lock()
client: httpx.AsyncClient | None = None


def conversation_key(body: dict) -> str | None:
    """Stable across every step of one agent trajectory, distinct between them."""
    msgs = body.get("messages")
    if not msgs:
        return None
    seed = json.dumps([m.get("content") for m in msgs[:2]], sort_keys=True)
    return hashlib.sha1(seed.encode()).hexdigest()


async def pick_backend(key: str | None) -> int:
    async with lock:
        if key is not None and key in affinity:
            idx = affinity[key]
        else:
            # Least in-flight; ties go to the lower index, which is fine because
            # the counts here are exact rather than a stale snapshot.
            idx = min(range(len(BACKENDS)), key=lambda i: inflight[i])
            if key is not None:
                affinity[key] = idx
        inflight[idx] += 1
        return idx


async def release(idx: int) -> None:
    async with lock:
        inflight[idx] -= 1


@app.get("/health")
async def health():
    return {"status": "ok", "backends": len(BACKENDS), "inflight": inflight}


@app.get("/router/stats")
async def stats():
    return {
        "inflight": {BACKENDS[i]: inflight[i] for i in range(len(BACKENDS))},
        "total_inflight": sum(inflight),
        "conversations_pinned": len(affinity),
    }


@app.get("/v1/models")
async def models():
    assert client is not None
    r = await client.get(f"{BACKENDS[0]}/v1/models")
    return Response(
        content=r.content, status_code=r.status_code, media_type="application/json"
    )


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def proxy(request: Request):
    assert client is not None
    raw = await request.body()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    idx = await pick_backend(conversation_key(body))
    url = f"{BACKENDS[idx]}{request.url.path}"
    # httpx sets no Content-Type when handed a raw `content=` body, and vLLM's
    # endpoint hard-rejects anything that isn't application/json with a 415.
    headers = {"Content-Type": "application/json"}
    if auth := request.headers.get("authorization"):
        headers["Authorization"] = auth

    if not body.get("stream"):
        try:
            r = await client.post(url, content=raw, headers=headers, timeout=TIMEOUT)
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )
        except Exception as e:
            log.warning("backend %s failed: %s", BACKENDS[idx], e)
            return JSONResponse({"error": str(e)}, status_code=502)
        finally:
            await release(idx)

    # Streaming: hold the slot open until the last chunk is forwarded, otherwise
    # in-flight counts would undercount and the balancing would be wrong.
    async def stream():
        try:
            req = client.build_request(
                "POST", url, content=raw, headers=headers, timeout=TIMEOUT
            )
            resp = await client.send(req, stream=True)
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
        except Exception as e:
            log.warning("backend %s stream failed: %s", BACKENDS[idx], e)
        finally:
            await release(idx)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(
        timeout=TIMEOUT,
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
    )
    log.info("routing over %d backends: %s", len(BACKENDS), ", ".join(BACKENDS))


@app.on_event("shutdown")
async def shutdown():
    if client is not None:
        await client.aclose()


if __name__ == "__main__":
    if not BACKENDS:
        sys.exit("set BACKENDS=http://host:port,http://host:port,...")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
