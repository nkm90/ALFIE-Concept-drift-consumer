import logging
from collections.abc import Callable

from jinja2 import Environment, FileSystemLoader

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_jinja_path = get_settings().jinja_path
jinja_environment = Environment(loader=FileSystemLoader(_jinja_path))


def render_deployment_instructions(renderer: Callable[[], str]) -> str | None:
    """Render deployment instructions, returning None instead of raising.

    Used by the training endpoints so a template failure never turns a
    successful training run into an error response.
    """
    try:
        return renderer()
    except Exception as e:
        logger.warning("Failed to render deployment instructions: %s", e)
        return None


def render_template(
    jinja_environment: Environment, template_name: str, **kwargs
) -> str:
    """
    Renders a Jinja2 template with the given arguments.
    """
    try:
        template = jinja_environment.get_template(template_name)
    except Exception as e:
        logger.error("Failed to load Jinja2 template '%s': %s", template_name, str(e))
        raise RuntimeError(f"Failed to load template '{template_name}': {e}") from e

    try:
        return template.render(**kwargs)
    except Exception as e:
        logger.error(
            "Failed to render Jinja2 template '%s' with kwargs %s: %s",
            template_name,
            kwargs,
            str(e),
        )
        raise RuntimeError(f"Failed to render template '{template_name}': {e}") from e
