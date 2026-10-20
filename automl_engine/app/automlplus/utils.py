"""Shared helpers for the AutoML+ service (image conversion, HTML, JSON)."""

import base64
import logging
import os
from io import BytesIO
from typing import Any

import requests
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
from PIL import Image

from app.core.exceptions import (
    AutoDWDownloadError,
    AutoMLImageError,
    AutoMLValidationError,
)
from app.core.config import get_settings
from app.core.utils import render_template

logger = logging.getLogger(__name__)
_jinja_path = get_settings().jinja_path

jinja_environment = Environment(loader=FileSystemLoader(_jinja_path))


def automl_plus_data_instructions() -> str:
    """Return the instructions for what kind of data is accepted by AutoML+."""
    if jinja_environment is not None:
        try:
            return render_template(jinja_environment, "automl_plus_accepted_format.md")
        except Exception as e:
            logger.error(f"Failed to render accepted format instructions: {e}")
            return "No accepted format instructions available"
    else:
        logger.warning("jinja_environment is None, returning default formats")
        return "Ask the agent for help"


class ImageConverter:
    """Convert images to base64 from local paths or URLs."""

    @staticmethod
    def to_base64(image_path_or_url: str) -> str:
        """Load an image from a local path or URL and return it as base64 PNG.

        Fetches over HTTP when the input looks like a URL (checking the
        content type really is an image), otherwise reads from disk. Either
        way the image is re-encoded as PNG before base64 encoding.
        """
        logger.info("Converting image to base64: %s", image_path_or_url)
        if not image_path_or_url or not isinstance(image_path_or_url, str):
            raise AutoMLValidationError(
                f"image_path_or_url must be a non-empty string, got: {type(image_path_or_url)}"
            )

        try:
            if image_path_or_url.startswith("http"):
                headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageConverter/1.0)"}
                resp = requests.get(image_path_or_url, headers=headers, timeout=30)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type:
                    raise AutoMLValidationError(
                        f"URL does not point to an image: {image_path_or_url} (Content-Type: {content_type})"
                    )
                image: Image.Image = Image.open(BytesIO(resp.content))
            else:
                if not os.path.isfile(image_path_or_url):
                    raise AutoMLValidationError(f"No such file: {image_path_or_url}")
                image = Image.open(image_path_or_url)

            image = image.convert("RGBA")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except requests.RequestException as e:
            logger.exception("Failed to fetch image from URL: %s", image_path_or_url)
            raise AutoDWDownloadError(f"Failed to fetch image from URL: {e}") from e
        except (IOError, OSError) as e:
            logger.exception("Failed to read image file: %s", image_path_or_url)
            raise AutoMLImageError(f"Failed to read image file: {e}") from e
        except Exception as e:
            logger.exception("Image conversion failed")
            raise AutoMLImageError(f"Image conversion failed: {e}") from e

    @staticmethod
    def bytes_to_base64(image_bytes: bytes) -> str:
        """Convert raw image bytes to base64 PNG string."""
        if not image_bytes or not isinstance(image_bytes, (bytes, bytearray)):
            raise AutoMLValidationError(
                f"image_bytes must be non-empty bytes, got: {type(image_bytes)}"
            )

        try:
            image = Image.open(BytesIO(image_bytes))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except (IOError, OSError) as e:
            logger.exception("Failed to decode image bytes")
            raise AutoMLImageError(f"Invalid image data: {e}") from e
        except Exception as e:
            logger.exception("Image bytes conversion failed")
            raise AutoMLImageError(f"Image bytes conversion failed: {e}") from e


def extract_text_from_html_bytes(content: bytes) -> str:
    """Extract readable text from raw HTML bytes."""
    try:
        if not content:
            logger.warning("Empty content provided to extract_text_from_html_bytes")
            return ""

        soup = BeautifulSoup(content, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        lines = (line.strip() for line in soup.get_text().splitlines())
        phrases = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in phrases if chunk)
        return text
    except Exception as e:
        logger.exception("Failed to extract text from HTML: %s", e)
        return ""


def json_safe(data: Any) -> Any:
    """Recursively convert string values to JSON-safe strings."""
    try:
        if isinstance(data, dict):
            return {k: json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [json_safe(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(json_safe(v) for v in data)
        elif isinstance(data, str):
            return (
                data.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
        elif isinstance(data, (int, float, bool, type(None))):
            return data
        else:
            return str(data)
    except Exception as e:
        logger.warning("Failed to make data JSON-safe: %s", e)
        return str(data)
