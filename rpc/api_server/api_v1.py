import logging
from typing import List
from sqlalchemy.orm import joinedload

from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import HTTPException

from deepfake import __version__
from deepfake.persistence import Video
from deepfake.persistence.data_model import User
from deepfake.rpc import RPC

from deepfake.rpc.api_server.api_auth import get_current_user
from deepfake.rpc.api_server.api_schemas import DeepfakeResponse, Ping, UserOut
from deepfake.rpc.api_server.deps import get_rpc
from deepfake.rpc.rpc import RPCException

logger = logging.getLogger(__name__)

API_VERSION = 2.43


# Public API, requires no auth.
router_public = APIRouter()
# Private API, protected by authentication
router = APIRouter()


@router_public.get("/ping", response_model=Ping)
def ping():
    """simple ping"""
    return {"status": "pong"}


@router.get("/deepfakes", tags=["info"],)
def deepfakes(rpc: RPC = Depends(get_rpc), user: UserOut = Depends(get_current_user)):
    try:
        return rpc._rpc_deepfakes(user)
    except RPCException:
        return []

@router.get("/deepfakes/{deepfake_id}", tags=["info"])
def get_deepfake_by_id(deepfake_id: str, rpc: RPC = Depends(get_rpc), user: UserOut = Depends(get_current_user)):
   
    try:
        return rpc._rpc_deepfake_by_id(user, deepfake_id)
    except RPCException:
        return []

@router.delete("/deepfakes/{deepfake_id}", tags=["delete"])
def delete_deepfake_endpoint(deepfake_id: str,  rpc: RPC = Depends(get_rpc), user: UserOut = Depends(get_current_user)):
    success = rpc._rpc_delete_deepfake(user, deepfake_id)
    if not success:
        raise HTTPException(status_code=404, detail="Deepfake not found")
    return {"message": "Deepfake deleted successfully"}