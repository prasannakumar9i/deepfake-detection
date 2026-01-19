"""
This module contains class to define RPC communications for deepfake operations.
"""

import logging
from abc import abstractmethod
from pathlib import Path
from sqlalchemy.orm import joinedload

from deepfake import __version__
from deepfake.constants import Config
from deepfake.rpc.api_server.api_schemas import UserOut
from deepfake.rpc.rpc_types import RPCSendMsg
from deepfake.persistence import Video, Result, Image
from deepfake.deepfake import DeepFake

logger = logging.getLogger(__name__)


class RPCException(Exception):
    """
    Raised with a rpc-formatted message in an _rpc_* method
    if the required state is wrong.
    """

    def __init__(self, message: str) -> None:
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message

    def __json__(self):
        return {"msg": self.message}


class RPCHandler:
    def __init__(self, rpc: "RPC", config: Config) -> None:
        self._rpc = rpc
        self._config: Config = config

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup pending module resources"""

    @abstractmethod
    def send_msg(self, msg: RPCSendMsg) -> None:
        """Sends a message to all registered RPC modules"""


class RPC:
    """
    RPC class provides access to database operations and deepfake analysis features.
    """

    def __init__(self, deepfake: DeepFake) -> None:
        self._deepfake = deepfake
        self._config = deepfake.config

    def _rpc_deepfakes(self, user: "UserOut"):
        """
        Fetch all stored deepfake entries (videos + images) for a given user.
        Returns a single merged list with a 'type' attribute added to each object.
        """
        videos = (
            Video.session.query(Video)
            .join(Video.result)
            .options(joinedload(Video.result))
            .filter(Video.user_id == user.id)
            .all()
        )

        images = (
            Image.session.query(Image)
            .join(Image.result)
            .options(joinedload(Image.result))
            .filter(Image.user_id == user.id)
            .all()
        )

        for v in videos:
            setattr(v, "type", "video")
        for i in images:
            setattr(i, "type", "image")

        return videos + images

    def _rpc_deepfake_by_id(self, user: "UserOut", deepfake_id: int):
        """
        Fetch a single deepfake entry by its ID.
        Searches both Video and Image tables, returns the first match.
        """
        video = (
            Video.session.query(Video)
            .options(joinedload(Video.result))
            .filter(Video.user_id == user.id, Video.id == deepfake_id)
            .one_or_none()
        )

        if video:
            setattr(video, "type", "video")
            return video

        image = (
            Image.session.query(Image)
            .options(joinedload(Image.result))
            .filter(Image.user_id == user.id, Image.id == deepfake_id)
            .one_or_none()
        )

        if image:
            setattr(image, "type", "image")

        return image

    def _rpc_image_analysis(self, file_path: str) -> Result:
        """
        Run deepfake analysis on an image and return a Result instance.
        """
        normalized_path = str(Path(file_path).resolve())
        model_name = self._config["model_name"]

        _result = self._deepfake.predictor.predict_image(normalized_path)
        label = _result["label"]

        return Result(
            analysis_model=model_name,
            deepfake_detected=(label == "Fake"),
            confidence=float(_result["confidence"]),
            detection_score=float(_result["scores"]["fake"]),
            real_score=float(_result["scores"]["real"]),
            fake_score=float(_result["scores"]["fake"]),
        )

    def _rpc_video_analysis(self, file_path: str) -> Result:
        """
        Run deepfake analysis on a video and return a Result instance.
        """
        normalized_path = str(Path(file_path).resolve())
        model_name = self._config["model_name"]

        _result = self._deepfake.predictor.predict_video(normalized_path)
        label = _result["label"]

        return Result(
            analysis_model=model_name,
            deepfake_detected=(label == "Fake"),
            confidence=float(_result["confidence"]),
            detection_score=float(_result["scores"]["fake"]),
            real_score=float(_result["scores"]["real"]),
            fake_score=float(_result["scores"]["fake"]),
        )

    def _rpc_add_deepfake_image(
        self,
        title: str,
        user_id: int,
        description: str,
        file_path: str,
        image_filename: str,
    ) -> dict:
        """
        Add a new deepfake image with analysis result.
        """
        try:
            result = self._rpc_image_analysis(file_path)
            image = Image(
                title=title,
                user_id=user_id,
                description=description,
                file_path=file_path,
                image_filename=image_filename,
                result=result,
            )

            Image.session.add(image)
            Image.session.commit()

            return {
                "id": image.id,
                "image_filename": image.image_filename,
            }

        except Exception as e:
            Image.session.rollback()
            logger.error(f"Failed to add deepfake image for user {user_id}: {e}")
            raise RuntimeError(f"Failed to add deepfake image: {str(e)}")

    def _rpc_add_deepfake_video(
        self,
        title: str,
        user_id: int,
        description: str,
        duration: str,
        file_path: str,
        video_filename: str,
    ) -> dict:
        """
        Add a new deepfake video with analysis result.
        """
        try:
            result = self._rpc_video_analysis(file_path)
            video = Video(
                title=title,
                user_id=user_id,
                description=description,
                duration=float(duration),
                file_path=file_path,
                video_filename=video_filename,
                result=result,
            )

            Video.session.add(video)
            Video.session.commit()

            return {
                "id": video.id,
                "video_filename": video.video_filename,
            }

        except Exception as e:
            Video.session.rollback()
            logger.error(f"Failed to add deepfake video for user {user_id}: {e}")
            raise RuntimeError(f"Failed to add deepfake video: {str(e)}")

    def _rpc_delete_deepfake(self, user: "UserOut", deepfake_id: str) -> bool:
        try:
            session = Video.session  # Use shared or injected session properly here

            video = (
                session.query(Video)
                .options(joinedload(Video.result))
                .filter(Video.id == deepfake_id, Video.user_id == user.id)
                .first()
            )

            if video:
                video.result = None  # Break relationship so cascade works
                session.delete(video)
                session.commit()
                return True

            # Try image similarly
            image = (
                Image.session.query(Image)
                .options(joinedload(Image.result))
                .filter(Image.id == deepfake_id, Image.user_id == user.id)
                .first()
            )

            if image:
                image.result = None
                Image.session.delete(image)
                Image.session.commit()
                return True

            return False

        except Exception as e:
            logger.error(
                f"Failed to delete deepfake entry {deepfake_id} for user {user.id}: {e}"
            )
            try:
                Video.session.rollback()
            except Exception:
                pass
            try:
                Image.session.rollback()
            except Exception:
                pass
            raise RuntimeError(f"Failed to delete deepfake: {str(e)}")

