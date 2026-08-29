# Modal reverse relay for a private-node vLLM

Harbor runs mini-swe-agent **inside** the Modal sandbox, so the agent calls the
deliberator over the network. A vLLM on a private Slurm node can dial out but
can't accept inbound, so Modal sandboxes can't reach it directly (the smoke test
failed with `OpenAIException - Connection error`). This relay is the reverse
tunnel that fixes it — entirely inside your own Modal account, no third-party
service:

```
Modal agent ──HTTP /v1/*──► relay (modal_relay.py) ──WS──► bridge.py ──► local vLLM
                             1 pinned warm container      ▲ node dials OUT and holds open
```

`mini-swe-agent` never streams, so this is a plain request/response relay.

## One-time deploy

```bash
# 1. shared secret (any random string), stored as a Modal secret
modal secret create tts-relay RELAY_SECRET=$(openssl rand -hex 16)

# 2. deploy the relay (stable URL, unlike an ngrok/quick-tunnel)
modal deploy harbor/relay/modal_relay.py
#   -> https://<workspace>--tts-vllm-relay-web.modal.run

# 3. record both in .env so the sbatch picks them up
echo 'RELAY_URL=https://<workspace>--tts-vllm-relay-web.modal.run' >> .env
echo 'RELAY_SECRET=<the same secret value>'                        >> .env
```

The relay pins one always-warm container (the node holds a single WebSocket to
it). **Stop it when you're done evaluating** so it isn't running idle:

```bash
modal app stop tts-vllm-relay
```

## Use

Nothing extra to do per run — `scripts/eval/harbor_eval.sbatch` detects `RELAY_URL`
+ `RELAY_SECRET` and, when it serves vLLM, automatically starts `bridge.py`
(dialing out to the relay) and points the eval's deliberator at
`${RELAY_URL}/v1`. Confirm health any time:

```bash
curl -s https://<workspace>--tts-vllm-relay-web.modal.run/health
# {"ok":true,"bridge_connected":true}   <- true once a node bridge is attached
```

## Pieces

- `modal_relay.py` — the Modal ASGI app: public `/v1/*` (agents) + `/bridge`
  WebSocket (the node). Auth: `Authorization: Bearer $RELAY_SECRET` on both.
- `bridge.py` — runs on the vLLM node; dials the relay's `/bridge`, forwards each
  request to `http://localhost:$VLLM_PORT`, reconnects on drop.

## Limits / notes

- The relay container is pinned (`min=max=1`) and kept warm; that's a small
  ongoing Modal cost while deployed — stop the app between runs.
- Request timeout is 600s (relay) / 900s (container) — generous for a single
  completion; raise in `modal_relay.py` if you use very long generations.
- Secret lives in the Modal secret + `.env`; it's never printed by the sbatch.
