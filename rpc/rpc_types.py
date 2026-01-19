from datetime import datetime
from typing import Any, Literal, TypedDict

ProfitLossStr = Literal["profit", "loss"]


class RPCSendMsgBase(TypedDict):
    pass
    # ty1pe: Literal[RPCMessageType]


RPCSendMsg = (
    RPCSendMsgBase
)
