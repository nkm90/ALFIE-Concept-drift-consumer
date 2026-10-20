"""VLM (Vision Language Model) tools for AutoML+.

A VLM task involves passing one or more images together with a text prompt to a
multimodal language model and processing its response. The classes here cover two
use-cases:

- ``ImagePromptRunner`` — general-purpose: run or stream any user-supplied prompt
  over an image (file upload or URL).
- ``AltTextChecker`` — specialised: evaluate whether provided alt text accurately
  describes an image, using a structured VLM prompt defined in Jinja2 templates.
"""

import logging

from dotenv import find_dotenv, load_dotenv
from jinja2 import Environment

from app.automlplus.utils import ImageConverter
from app.core.chat_handler import ChatHandler
from app.core.config import get_settings
from app.core.exceptions import AutoMLChatError, AutoMLImageError, AutoMLValidationError
from app.core.utils import render_template

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


class ImagePromptRunner:
    """Run a VLM on an image and user-provided prompt."""

    DEFAULT_MODEL: str = get_settings().image_prompt_model

    @staticmethod
    def _resolve_model(model: str | None) -> str:
        if not model or not str(model).strip():
            return ImagePromptRunner.DEFAULT_MODEL
        return model

    @staticmethod
    def build_messages(
        jinja_environment: Environment | None, image_b64: str, prompt: str
    ) -> list[dict[str, str | list[str] | list[None]]]:
        messages: list[dict[str, str | list[str] | list[None]]] = []
        if jinja_environment is not None:
            try:
                system_prompt = render_template(
                    jinja_environment, "image_to_website_prompt.txt"
                )
                messages.append({"role": "system", "content": system_prompt})
            except Exception:
                raise
        messages.append({"role": "user", "content": prompt, "images": [image_b64]})
        return messages

    @staticmethod
    def run(
        image_bytes: bytes | None = None,
        image_path_or_url: str | None = None,
        prompt: str = "",
        model: str | None = None,
        jinja_environment: Environment | None = None,
    ) -> str:
        model_name = ImagePromptRunner._resolve_model(model)
        try:
            if image_bytes is None and not image_path_or_url:
                raise AutoMLValidationError(
                    "Provide either image_bytes or image_path_or_url"
                )

            image_b64 = (
                ImageConverter.bytes_to_base64(image_bytes)
                if image_bytes is not None
                else ImageConverter.to_base64(str(image_path_or_url))
            )

            messages = ImagePromptRunner.build_messages(
                jinja_environment, image_b64, prompt
            )

            return ChatHandler.chat_sync_messages(messages=messages, model=model_name)
        except Exception:
            logger.exception("ImagePromptRunner failed")
            raise

    @staticmethod
    def run_stream(
        image_bytes: bytes | None = None,
        image_path_or_url: str | None = None,
        prompt: str = "",
        model: str | None = None,
        jinja_environment: Environment | None = None,
    ):
        """Stream VLM output for an image+prompt interaction. Yields incremental text chunks."""
        model_name = ImagePromptRunner._resolve_model(model)

        if image_bytes is None and not image_path_or_url:
            raise AutoMLValidationError(
                "Provide either image_bytes or image_path_or_url"
            )

        try:
            image_b64 = (
                ImageConverter.bytes_to_base64(image_bytes)
                if image_bytes is not None
                else ImageConverter.to_base64(str(image_path_or_url))
            )
        except Exception as e:
            logger.exception("Failed to convert image to base64 in run_stream")
            raise AutoMLImageError(f"Image conversion failed: {e}") from e

        try:
            messages = ImagePromptRunner.build_messages(
                jinja_environment, image_b64, prompt
            )
        except Exception as e:
            logger.exception("Failed to build messages in run_stream")
            raise AutoMLChatError(f"Message building failed: {e}") from e

        try:
            return ChatHandler.chat_stream_messages_sync(
                messages=messages, model=model_name
            )
        except Exception as e:
            logger.exception("Failed to start streaming in run_stream")
            raise AutoMLChatError(f"Streaming failed: {e}") from e


class AltTextChecker:
    """Check whether provided alt text matches an image using a VLM."""

    DEFAULT_MODEL = "gpt-4o-mini"

    @staticmethod
    def _resolve_model(model: str) -> str:
        """Return a valid model string, normalizing common aliases and falling back to default."""
        if not model or model.strip() == "":
            logger.error(
                "Model parameter is empty or None, using default '%s'",
                AltTextChecker.DEFAULT_MODEL,
            )
            return AltTextChecker.DEFAULT_MODEL

        candidate = model.strip()
        lower = candidate.lower().replace(" ", "")

        if lower in {"gpt40-mini", "gpt4o-mini"}:
            return "gpt-4o-mini"

        return lower

    @staticmethod
    def _build_messages(
        jinja_environment: Environment, image_b64: str, alt_text: str
    ) -> list[dict]:
        """Construct the message payload for the VLM call."""
        return [
            {
                "role": "system",
                "content": render_template(
                    jinja_environment, "wcag_checker_default_prompt.txt"
                ),
            },
            {"role": "user", "content": f"Alt text: {alt_text}"},
            {
                "role": "user",
                "content": render_template(
                    jinja_environment, "image_alt_checker_prompt.txt"
                ),
                "images": [image_b64],
            },
        ]

    @staticmethod
    def _redact_messages_for_log(messages: list[dict]) -> list[dict]:
        """Return a copy of messages with any base64 image payloads redacted for logging."""
        redacted: list[dict] = []
        for message in messages:
            msg_copy = {k: v for k, v in message.items() if k != "images"}
            if "images" in message:
                safe_images = []
                for img in message["images"]:
                    length_hint = len(img) if isinstance(img, str) else None
                    safe_images.append(
                        f"<redacted_base64 length={length_hint}>"
                        if length_hint is not None
                        else "<redacted_base64>"
                    )
                msg_copy["images"] = safe_images
            redacted.append(msg_copy)
        return redacted

    @staticmethod
    def check(
        jinja_environment: Environment,
        image_url_or_path: str,
        alt_text: str,
        model: str = get_settings().alt_text_checker_model or DEFAULT_MODEL,
    ) -> str:
        logger.info("Checking alt-text using model %s", model)
        model = AltTextChecker._resolve_model(model)
        messages = None

        try:
            image_b64 = ImageConverter.to_base64(image_url_or_path)

            messages = AltTextChecker._build_messages(
                jinja_environment=jinja_environment,
                image_b64=image_b64,
                alt_text=alt_text,
            )

            logger.info("Sending request with model: %s", model)
            logger.info(
                "Messages structure (redacted): %s",
                AltTextChecker._redact_messages_for_log(messages),
            )

            response_content = ChatHandler.chat_sync_messages(
                messages=messages,
                model=model,
            )

            return response_content

        except Exception as e:
            logger.exception("AltTextChecker failed with error: %s", str(e))
            logger.error("Model used: %s", model)
            if messages is not None:
                try:
                    logger.error(
                        "Messages sent (redacted): %s",
                        AltTextChecker._redact_messages_for_log(messages),
                    )
                except Exception:
                    logger.error("Messages sent (redaction_failed)")
            raise
