from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from deepfake.persistence import Video, Image
from deepfake.persistence.models import _request_id_ctx_var
from deepfake.rpc.rpc import RPC

from .webserver import ApiServer

def get_rpc_optional() -> RPC | None:
    if ApiServer._has_rpc:
        return ApiServer._rpc
    return None

async def get_rpc() -> AsyncIterator[RPC]:
    _rpc = get_rpc_optional()
    if _rpc:
        request_id = str(uuid4())
        ctx_token = _request_id_ctx_var.set(request_id)
        Video.rollback()
        Image.rollback()
        try:
            yield _rpc
        finally:
            Video.session.remove()
            _request_id_ctx_var.reset(ctx_token)

    else:
        raise Exception("Bot is not in the correct state")


def get_config() -> dict[str, Any]:
    return ApiServer._config


def get_api_config() -> dict[str, Any]:
    return ApiServer._config["api_server"]

