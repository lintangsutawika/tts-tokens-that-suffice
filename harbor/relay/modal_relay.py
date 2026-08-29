"""
Modal-hosted reverse relay so a private GPU node's vLLM is reachable from Modal.

The problem: Harbor runs mini-swe-agent *inside* a Modal sandbox, and the agent
calls the deliberator over the network. A vLLM on a private Slurm node can dial
OUT but cannot accept inbound, so Modal sandboxes can't reach it directly.

This relay is the public rendezvous: it exposes an OpenAI-compatible HTTPS
endpoint (what the agent calls) and a WebSocket the node dials out to and holds
open (harbor/relay/bridge.py). Requests arriving on HTTP are forwarded over the
WebSocket to the node, which proxies them to its local vLLM and returns the
response. Everything lives in *your* Modal account — no third-party tunnel.

    Modal agent ──HTTP /v1/*──► relay (this app) ──WS──► node bridge ──► local vLLM
                                 (1 pinned warm container)   ▲ node dials out

mini-swe-agent never streams (plain litellm.completion), so a simple
request/response relay suffices — no SSE plumbing.

Deploy once (stable URL):
    modal secret create tts-relay RELAY_SECRET=$(openssl rand -hex 16)
    modal deploy harbor/relay/modal_relay.py
    # -> https://<workspace>--tts-vllm-relay-web.modal.run
Stop it when idle (the pinned container otherwise stays warm):
    modal app stop tts-vllm-relay
"""

import asyncio
import json
import os
import uuid

import modal

# App and secret names are env-overridable so you can deploy a SECOND, independent
# relay for a concurrent run with a different model (the relay is a hard singleton:
# one warm container holding one bridge socket, so two nodes on the same app fight
# over it — last connector wins all traffic). Deploy a second instance with:
#   RELAY_APP_NAME=tts-vllm-relay-b RELAY_SECRET_NAME=tts-relay-b \
#       modal deploy harbor/relay/modal_relay.py
# -> https://<workspace>--tts-vllm-relay-b-web.modal.run  (own URL, own secret)
# harbor_eval.sbatch's EPHEMERAL_RELAY=1 mode uses this to give each Slurm job its
# own relay (RELAY_APP_NAME=tts-vllm-relay-<jobid>), torn down on job exit.
APP_NAME = os.environ.get("RELAY_APP_NAME", "tts-vllm-relay")
SECRET_NAME = os.environ.get("RELAY_SECRET_NAME", "tts-relay")

# Secret source: an inline RELAY_SECRET present in the DEPLOY environment is baked
# into the app (Secret.from_dict) — this is what per-job ephemeral relays use, so
# no `modal secret create` step is needed. Otherwise fall back to a named Modal
# secret (the stable shared relay uses `tts-relay`).
if os.environ.get("RELAY_SECRET"):
    _relay_secret = modal.Secret.from_dict({"RELAY_SECRET": os.environ["RELAY_SECRET"]})
else:
    _relay_secret = modal.Secret.from_name(SECRET_NAME)

app = modal.App(APP_NAME)

image = modal.Image.debian_slim().pip_install("fastapi==0.115.*", "websockets>=12")


@app.function(
    image=image,
    # Pin to exactly one always-warm container: the node holds a single WebSocket
    # to it, and every HTTP request must land on that same container to be
    # forwarded over that socket.
    min_containers=1,
    max_containers=1,
    # Idle scaledown window. Bounds how long the container lingers after its last
    # request — the safety net for a relay orphaned by a hard SIGKILL (app never
    # `stop`ped). NOTE: min_containers=1 pins one container warm while the app stays
    # deployed, so this only takes effect once the app is stopped/undeployed; drop
    # min_containers to 0 if you want an orphaned app to truly self-reap on idle.
    scaledown_window=600,
    # A WebSocket is one long-lived ASGI request, so `timeout` caps its lifetime.
    # At 900s the node bridge was force-dropped every 15 min, and any request
    # in flight at that moment was lost (agent stalls the relay's 600s wait ->
    # AgentTimeoutError). Give the socket a full day so it spans an eval run.
    timeout=86400,
    # This single container's ONE event loop fans in every agent's traffic (up to
    # max_inputs concurrent HTTP proxies + the bridge WebSocket). On the default
    # fractional CPU it could not keep up under load: the loop stalled shuttling
    # large SWE-bench payloads and missed the bridge's WebSocket keepalive, so the
    # bridge tore the socket down (1011) and every in-flight request was lost.
    # Give the loop real cores so it can pong on time and forward without stalling.
    cpu=4.0,
    secrets=[_relay_secret],
)
@modal.concurrent(max_inputs=64)  # many concurrent agents multiplexed over one WS
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException, Request, Response, WebSocket

    api = FastAPI()
    secret = os.environ["RELAY_SECRET"]
    # Per-container state (safe: exactly one container).
    bridge_ws: dict[str, object] = {"ws": None}
    pending: dict[str, asyncio.Future] = {}
    # Up to `max_inputs` proxy handlers share the one bridge WebSocket. Starlette's
    # send_text is not concurrency-safe: unserialized writes interleave frames on
    # the wire, the peer reads a corrupt frame and drops the socket with a 1002
    # protocol error. Serialize every send through one lock.
    send_lock = asyncio.Lock()

    @api.websocket("/bridge")
    async def bridge(ws: WebSocket):
        # Auth the node before accepting.
        if ws.headers.get("authorization") != f"Bearer {secret}":
            await ws.close(code=1008)
            return
        await ws.accept()
        bridge_ws["ws"] = ws
        try:
            while True:
                raw = await ws.receive_text()
                # Deserialize the (large) response body off the loop so the loop
                # stays free to service other proxies and answer keepalive pings.
                frame = await asyncio.to_thread(json.loads, raw)
                fut = pending.pop(frame["id"], None)
                if fut is not None and not fut.done():
                    fut.set_result(frame)
        except Exception:
            pass
        finally:
            if bridge_ws["ws"] is ws:
                bridge_ws["ws"] = None

    @api.get("/health")
    async def health():
        return {"ok": True, "bridge_connected": bridge_ws["ws"] is not None}

    @api.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if request.headers.get("authorization") != f"Bearer {secret}":
            raise HTTPException(status_code=401, detail="bad relay token")
        ws = bridge_ws["ws"]
        if ws is None:
            raise HTTPException(status_code=503, detail="no vLLM bridge connected")
        rid = uuid.uuid4().hex
        frame = {
            "id": rid,
            "method": request.method,
            "path": f"/v1/{path}",
            "query": request.url.query or "",
            "body": (await request.body()).decode(),
        }
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        pending[rid] = fut
        # Serialize the (large) request frame off the loop before taking the send
        # lock -- a synchronous json.dumps of a ~100s-of-KB prompt body would block
        # the loop (and every other proxy + the keepalive) while it ran.
        payload = await asyncio.to_thread(json.dumps, frame)
        async with send_lock:
            await ws.send_text(payload)
        try:
            resp = await asyncio.wait_for(fut, timeout=600)
        except asyncio.TimeoutError:
            pending.pop(rid, None)
            raise HTTPException(status_code=504, detail="relay timeout waiting for node")
        return Response(
            content=resp["body"],
            status_code=resp["status"],
            media_type=resp.get("content_type", "application/json"),
        )

    return api
