"""Chat handling utilities for async LLM requests.

Provides a static facade (`ChatHandler`) to interact with Azure AI Inference
models, supporting both regular and streaming responses.
"""

import asyncio
import logging
from typing import Any, List

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 4

# Lazy, config-keyed singleton for the Azure chat client. ``_get_azure_client``
# builds it once per (endpoint, api_key) pair and reuses it; call
# ``reset_azure_client`` to drop the cache (config change / tests).
_azure_client: ChatCompletionsClient | None = None
_azure_client_config: tuple[str, str] | None = None


def reset_azure_client() -> None:
    """Drop the cached Azure chat client."""
    global _azure_client, _azure_client_config
    _azure_client = None
    _azure_client_config = None


class ChatHandler:
    _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    @staticmethod
    async def chat(
        message, context="", backend="azure", model="gpt-4o-mini", stream=False
    ):
        """Send a chat request, limiting how many run at once.

        Wraps every call in a shared semaphore so at most four requests hit the
        backend concurrently. Returns the response text directly, or (when
        ``stream`` is set) an async generator that yields text chunks as they
        arrive.
        """
        async with ChatHandler._semaphore:
            if stream:

                async def stream_gen():
                    async for chunk in ChatHandler.dispatch_stream(
                        message, context, backend, model
                    ):
                        yield chunk

                return stream_gen()
            else:
                return await ChatHandler.dispatch(message, context, backend, model)

    @staticmethod
    async def dispatch(message, context, backend, model):
        """Route a non-streaming chat request to the correct backend."""
        logger.debug(f"Dispatch Chat to backend {backend} with {message}, {context}")
        if backend.lower() == "azure":
            return await ChatHandler._azure_chat(message, context, model)
        else:
            raise ValueError(f"Unknown chat backend: {backend}")

    @staticmethod
    async def dispatch_stream(message, context, backend, model):
        """Route a streaming chat request to the correct backend, yielding chunks."""
        logger.debug(
            f"Dispatch Chat Stream to backend {backend} with {message}, {context}"
        )
        if backend.lower() == "azure":
            async for chunk in ChatHandler._azure_chat_stream(message, context, model):
                yield chunk
        else:
            raise ValueError(f"Unknown chat backend: {backend}")

    # --- Synchronous helpers for structured message payloads (incl. images) ---
    @staticmethod
    def chat_sync_messages(
        messages: List[dict], backend: str = "azure", model: str = "gpt-4o-mini"
    ) -> str:
        """Synchronously send a list of chat messages (optionally with images) to Azure."""
        logger.debug(f"Dispatch synchronous Chat to backend {backend} with {messages}")
        try:
            if backend.lower() == "azure":
                return ChatHandler._azure_chat_messages_sync(messages, model)
            else:
                raise ValueError(f"Unknown chat backend: {backend}")
        except Exception as e:
            logger.error("Chat sync messages failed: %s", e)
            raise

    # --- Streaming helpers for structured message payloads (incl. images) ---
    @staticmethod
    def chat_stream_messages_sync(
        messages: List[dict], backend: str = "azure", model: str = "gpt-4o-mini"
    ):
        """Synchronously stream a list of chat messages (optionally with images).

        Yields incremental text chunks from Azure as they arrive.
        """
        logger.debug("Stream chat messages synchronously")
        try:
            if backend.lower() == "azure":
                return ChatHandler._azure_chat_messages_stream_sync(messages, model)
            else:
                raise ValueError(f"Unknown chat backend: {backend}")
        except Exception as e:
            logger.error("Chat stream messages sync failed: %s", e)
            raise

    @staticmethod
    def _azure_chat_messages_stream_sync(messages: List[dict], model: str):
        """Synchronously stream structured messages through Azure.

        Iterates over the stream events, pulling text out of each delta while
        tolerating the different shapes the SDK returns (strings, dicts, or
        content-part lists), and closes the stream when done.
        """
        client = ChatHandler._get_azure_client()
        azure_msgs = ChatHandler._to_azure_messages(messages)
        logger.debug("Azure client stream in chunks")
        stream = client.complete(model=model, messages=azure_msgs, stream=True)
        for event in stream:
            if hasattr(event, "delta") and event.delta:
                delta = event.delta
                content = getattr(delta, "content", None)
                if content is None and isinstance(delta, dict):
                    content = delta.get("content")
                if isinstance(content, str):
                    if content:
                        yield content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            if item:
                                yield item
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if text:
                                yield text
                        else:
                            text = getattr(item, "text", None)
                            if text:
                                yield text
        if hasattr(stream, "close"):
            stream.close()

    @staticmethod
    def _azure_chat_messages_sync(messages: List[dict], model: str) -> str:
        """Synchronous Azure chat that supports text and images.

        Accepts our internal message dicts of the form:
          {"role": "system"|"user", "content": str | None, "images": [base64str, ...]?}

        Converts them into Azure AI Inference message objects. For messages that
        include images, constructs a single UserMessage with a mixed content list
        containing a text item (when provided) and one input_image item per image.
        """
        client = ChatHandler._get_azure_client()
        logger.debug("Azure client init stream")
        azure_msgs = ChatHandler._to_azure_messages(messages)
        logger.debug(f"Azure message dict {azure_msgs}")
        response = client.complete(model=model, messages=azure_msgs)
        return ChatHandler._extract_azure_text_from_response(response)

    @staticmethod
    def _to_azure_messages(msgs: List[dict]) -> List[Any]:
        """Convert internal message dicts to Azure AI Inference message objects."""
        azure_messages: List[Any] = []

        for m in msgs:
            role = (m.get("role") or "user").lower()
            text_content = m.get("content")
            images = m.get("images") or []

            if not images:
                if role == "system":
                    azure_messages.append(SystemMessage(content=text_content or ""))
                else:
                    azure_messages.append(UserMessage(content=text_content or ""))
                continue

            mixed_content: List[Any] = []
            if text_content:
                mixed_content.append({"type": "text", "text": text_content})
            for b64 in images:
                if not isinstance(b64, str) or not b64:
                    continue
                mixed_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )

            # Azure doesn't accept images in system role; downgrade to user
            azure_messages.append(UserMessage(content=mixed_content))

        return azure_messages

    @staticmethod
    def _get_azure_client() -> ChatCompletionsClient:
        """Return a cached Azure AI Foundry chat client.

        The client is built once per (endpoint, api_key) pair and reused across
        calls so we don't reconstruct the HTTP client on every request. Call
        ``reset_azure_client`` to drop the cache.
        """
        global _azure_client, _azure_client_config
        settings = get_settings()
        endpoint = settings.azure_openai_endpoint_large_model
        api_key = settings.azure_openai_key
        if not endpoint or not api_key:
            raise RuntimeError(
                "Missing AZURE_OPENAI_ENDPOINT_LARGE_MODEL or AZURE_OPENAI_KEY "
                "environment variables"
            )
        config = (endpoint, api_key)
        if _azure_client is not None and _azure_client_config == config:
            return _azure_client
        logger.debug("Creating new Azure client for endpoint %s", endpoint)
        try:
            client = ChatCompletionsClient(
                endpoint=endpoint, credential=AzureKeyCredential(api_key)
            )
        except Exception as e:
            logger.error("Failed to initialize Azure client: %s", e)
            raise RuntimeError(f"Failed to initialize Azure client: {e}") from e
        _azure_client = client
        _azure_client_config = config
        return client

    @staticmethod
    async def _azure_chat(message, context, model):
        """Non-streaming chat call using Azure GPT-4o-mini."""
        client = ChatHandler._get_azure_client()
        messages = [
            SystemMessage(content=context or "You are a helpful assistant."),
            UserMessage(content=message),
        ]
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: client.complete(model=model, messages=messages)
            )
            logger.debug("Azure chat async works")
            return ChatHandler._extract_azure_text_from_response(response)
        except Exception as e:
            logger.error("Azure chat request failed: %s", e)
            raise

    @staticmethod
    async def _azure_chat_stream(message, context, model):
        """Streaming chat call using Azure GPT-4o-mini."""
        client = ChatHandler._get_azure_client()
        messages = [
            SystemMessage(content=context or "You are a helpful assistant."),
            UserMessage(content=message),
        ]

        loop = asyncio.get_running_loop()

        def sync_stream():
            return client.complete(
                model=model,
                messages=messages,
                stream=True,
            )

        # Run the sync generator in a thread and forward chunks asynchronously
        try:
            stream = await loop.run_in_executor(None, sync_stream)
            for event in stream:
                if hasattr(event, "delta") and event.delta:
                    delta = event.delta
                    content = getattr(delta, "content", None)
                    if content is None and isinstance(delta, dict):
                        content = delta.get("content")
                    # content may be str or list of items (with text)
                    if isinstance(content, str):
                        if content:
                            yield content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, str):
                                if item:
                                    yield item
                            elif isinstance(item, dict):
                                text = item.get("text")
                                if text:
                                    yield text
                            else:
                                text = getattr(item, "text", None)
                                if text:
                                    yield text
            logger.debug("Azure chat stream works")
        except Exception as e:
            logger.error("Azure chat stream request failed: %s", e)
            raise
        finally:
            if hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    pass

    @staticmethod
    def _extract_azure_text_from_response(response) -> str:
        """Extract text content from Azure ChatCompletions response across SDK variants."""
        # Try choices-based response (common structure)
        try:
            choice0 = response.choices[0]
            message = getattr(choice0, "message", None) or choice0["message"]
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            # content can be str or list of content parts
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        txt = part.get("text")
                        if txt:
                            parts.append(txt)
                    else:
                        txt = getattr(part, "text", None)
                        if txt:
                            parts.append(txt)
                return "".join(parts).strip()
        except Exception:
            pass
        # Fallback older shape: output_message.content[0].text
        try:
            return response.output_message.content[0].text.strip()
        except Exception:
            pass
        # Last resort: str(response)
        return str(response)
