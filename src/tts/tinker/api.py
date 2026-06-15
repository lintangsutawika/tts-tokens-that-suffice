"""
Extended skyrl tinker API server.

Adds /api/v1/client/config (feature-flag negotiation introduced in tinker>=0.18)
that skyrl 0.2.0 does not implement. All other routes are delegated to skyrl's app.

Run instead of skyrl.tinker.api:
    TINKER_API_KEY=tml-dummy uv run -m tts.tinker.api \
        --base-model "Qwen/Qwen3-4B-Instruct-2507" --backend fsdp
"""

import skyrl.tinker.api as _skyrl_api
from pydantic import BaseModel
from skyrl.tinker.api import app, add_model, EngineConfig, get_uvicorn_log_config

# Patch the sentinel so skyrl can parse our module path from the process cmdline
_skyrl_api.API_SERVER_STARTUP_ARGS = ["-m", "tts.tinker.api"]


class _ClientConfigRequest(BaseModel):
    sdk_version: str = ""


class _ClientConfigResponse(BaseModel):
    pjwt_auth_enabled: bool = False
    credential_default_source: str = "api_key"
    sample_dispatch_bytes_semaphore_size: int = 10 * 1024 * 1024
    inflight_response_bytes_semaphore_size: int = 50 * 1024 * 1024
    parallel_fwdbwd_chunks: bool = True
    proto_write_fwdbwd: bool = False
    proto_compress_fwdbwd: bool = False
    fwd_via_fwdbwd: bool = False
    billing_exception_max_pause_duration_sec: int = 3600
    sample_no_retries: bool = False
    sample_enable_stuck_detection: bool = True
    use_pyqwest_transport: bool = True


@app.post("/api/v1/client/config", response_model=_ClientConfigResponse)
async def client_config(request: _ClientConfigRequest) -> _ClientConfigResponse:
    return _ClientConfigResponse()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="SkyRL tinker API server")
    add_model(parser, EngineConfig)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app.state.engine_config = EngineConfig.model_validate(
        {k: v for k, v in vars(args).items() if k in EngineConfig.model_fields}
    )

    uvicorn.run(app, host=args.host, port=args.port, log_config=get_uvicorn_log_config())
