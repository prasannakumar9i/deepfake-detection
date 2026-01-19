import asyncio
import logging
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from fastapi.staticfiles import StaticFiles
import orjson
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from deepfake.constants import Config
from deepfake.exceptions import OperationalException
from deepfake.rpc.api_server.uvicorn_threaded import UvicornServer
from deepfake.rpc import RPC, RPCHandler
from deepfake.rpc.rpc_types import RPCSendMsg


logger = logging.getLogger(__name__)


class FTJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """
        Use rapidjson for responses
        Handles NaN and Inf / -Inf in a javascript way by default.
        """
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class ApiServer(RPCHandler):
    __instance = None
    __initialized = False

    _rpc: RPC
    _has_rpc: bool = False
    _config: Config = {}
  
    def __new__(cls, *args, **kwargs):
        """
        This class is a singleton.
        We'll only have one instance of it around.
        """
        if ApiServer.__instance is None:
            ApiServer.__instance = object.__new__(cls)
            ApiServer.__initialized = False
        return ApiServer.__instance

    def __init__(self, config: Config, standalone: bool = False) -> None:
        ApiServer._config = config
        if self.__initialized and (standalone or self._standalone):
            return
        self._standalone: bool = standalone
        self._server = None

        ApiServer.__initialized = True

        self.app = FastAPI(
            title="Freqtrade API",
            redoc_url=None,
            default_response_class=FTJSONResponse,
        )
        self.configure_app(self.app, self._config)
        self.start_api()

    def add_rpc_handler(self, rpc: RPC):
        """
        Attach rpc handler
        """
        if not ApiServer._has_rpc:
            ApiServer._rpc = rpc
            ApiServer._has_rpc = True
        else:
            # This should not happen assuming we didn't mess up.
            raise OperationalException("RPC Handler already attached.")

    def cleanup(self) -> None:
        """Cleanup pending module resources"""
        ApiServer._has_rpc = False
        del ApiServer._rpc

        if self._server and not self._standalone:
            logger.info("Stopping API Server")
            # self._server.force_exit, self._server.should_exit = True, True
            self._server.cleanup()

    @classmethod
    def shutdown(cls):
        cls.__initialized = False
        del cls.__instance
        cls.__instance = None
        cls._has_rpc = False
        cls._rpc = None


    def handle_rpc_exception(self, request, exc):
        logger.error(f"API Error calling: {exc}")
        return JSONResponse(
            status_code=502, content={"error": f"Error querying {request.url.path}: {exc.message}"}
        )

    def handle_unexpected_exception(self, request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred."}
        )


    def configure_app(self, app: FastAPI, config):
        from deepfake.rpc.api_server.api_auth import get_current_user, router_login
        from deepfake.rpc.api_server.api_background_tasks import router as api_bg_tasks
        from deepfake.rpc.api_server.api_v1 import router as api_v1
        from deepfake.rpc.api_server.api_v1 import router_public as api_v1_public
        from deepfake.rpc.api_server.web_ui import router_ui
        from deepfake.rpc.rpc import RPCException

        # Public
        app.include_router(api_v1_public, prefix="/api")

        # Auth
        app.include_router(router_login, prefix="/api", tags=["auth"])

        # Protected
        app.include_router(
            api_v1,
            prefix="/api",
        )
        
        app.include_router(
            api_bg_tasks,
            prefix="/api",
        )

        app.include_router(router_ui, prefix="", include_in_schema=False)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get("api_server", {}).get("CORS_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Exception Handlers
        app.add_exception_handler(RPCException, self.handle_rpc_exception)
        app.add_exception_handler(Exception, self.handle_unexpected_exception)  # Catch-all


    def start_api(self):
        """
        Start API ... should be run in thread.
        """
        rest_ip = self._config["api_server"]["listen_ip_address"]
        rest_port = self._config["api_server"]["listen_port"]

        logger.info(f"Starting HTTP Server at {rest_ip}:{rest_port}")
        if not ip_address(rest_ip).is_loopback:
            logger.warning("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.warning(
                "SECURITY WARNING - This is insecure please set to your loopback,"
                "e.g 127.0.0.1 in config.json"
            )

        logger.info("Starting Local Rest Server.")
        verbosity = self._config["api_server"].get("verbosity", "error")

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        uvconfig = uvicorn.Config(
            self.app,
            port=rest_port,
            host=rest_ip,
            use_colors=True,
            log_config=None,
            access_log=True if verbosity != "error" else False,
            ws_ping_interval=None,  # We do this explicitly ourselves
        )
        try:
            self._server = UvicornServer(uvconfig)
            if self._standalone:
                self._server.run()
            else:
                self._server.run_in_thread()
        except Exception:
            logger.exception("Api server failed to start.")
