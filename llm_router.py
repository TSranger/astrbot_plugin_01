import asyncio
import json
from typing import Any
from urllib import error, request

from astrbot.api import logger


class PluginLLMRouter:
    """Route plugin LLM calls by role without modifying AstrBot core.

    The router supports two modes:
    1. ``astrbot_default``: delegate all calls to the current AstrBot provider.
    2. ``plugin_router``: use plugin-local OpenAI-compatible endpoints per role.

    Args:
        context: AstrBot plugin context.
        settings: The ``llm_settings`` section from ``config.yaml``.
    """

    def __init__(self, context: Any, settings: dict[str, Any] | None):
        self.context = context
        self.settings = settings or {}
        self.mode = self.settings.get("mode", "astrbot_default")
        self.fallback_to_default = self.settings.get(
            "fallback_to_astrbot_default", True
        )
        self.roles = self.settings.get("roles", {})

    async def text_chat(
        self,
        role: str,
        prompt: str,
        system_prompt: str,
        context_messages: list[dict[str, str]] | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        """Generate text for a specific plugin role.

        Args:
            role: Logical role name such as ``chat`` or ``analysis``.
            prompt: User prompt content.
            system_prompt: System prompt content.
            context_messages: Optional chat history in OpenAI message format.
            image_urls: Optional list of image URLs for vision-capable models.

        Returns:
            Generated text content.
        """
        role_config = self.roles.get(role, {})
        if self.mode == "plugin_router" and role_config.get("enabled"):
            try:
                return await self._call_openai_compatible(
                    role_config,
                    prompt,
                    system_prompt,
                    context_messages or [],
                    image_urls=image_urls,
                )
            except Exception as exc:
                logger.error(
                    f"[LLM Router] Plugin router failed for role={role}: {exc}"
                )
                if not self.fallback_to_default:
                    raise
        return await self._call_astrbot_default(prompt, system_prompt)

    async def _call_astrbot_default(
        self,
        prompt: str,
        system_prompt: str,
    ) -> str:
        """Call the current AstrBot provider.

        Args:
            prompt: User prompt content.
            system_prompt: System prompt content.

        Returns:
            Generated text content.
        """
        provider = self.context.get_using_provider()
        if not provider:
            raise RuntimeError("AstrBot default provider is unavailable.")
        response = await provider.text_chat(
            prompt=prompt,
            system_prompt=system_prompt,
            context=[],
        )
        return getattr(
            response, "completion_text", getattr(response, "text", "")
        ).strip()

    async def _call_openai_compatible(
        self,
        role_config: dict[str, Any],
        prompt: str,
        system_prompt: str,
        context_messages: list[dict[str, str]],
        image_urls: list[str] | None = None,
    ) -> str:
        """Call an OpenAI-compatible endpoint defined in plugin config.

        Args:
            role_config: Role-specific OpenAI-compatible config.
            prompt: User prompt content.
            system_prompt: System prompt content.
            context_messages: Optional message history.
            image_urls: Optional list of image URLs for vision requests.

        Returns:
            Generated text content.
        """
        if role_config.get("provider_type") != "openai_compatible":
            raise ValueError(
                f"Unsupported provider_type: {role_config.get('provider_type')}",
            )

        base_url = str(role_config.get("base_url", "")).strip()
        model = str(role_config.get("model", "")).strip()
        if not base_url or not model:
            raise ValueError("base_url and model are required in plugin_router mode.")

        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_messages)

        if image_urls:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for url in image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": role_config.get("temperature", 0.7),
            "top_p": role_config.get("top_p", 1.0),
            "max_tokens": role_config.get("max_tokens", 512),
        }

        headers = {
            "Content-Type": "application/json",
        }
        api_key = str(role_config.get("api_key", "")).strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        extra_headers = role_config.get("extra_headers", {})
        if isinstance(extra_headers, dict):
            for key, value in extra_headers.items():
                headers[str(key)] = str(value)

        timeout_seconds = int(role_config.get("timeout_seconds", 60))
        response_data = await asyncio.to_thread(
            self._post_json,
            endpoint,
            headers,
            payload,
            timeout_seconds,
        )
        return self._extract_openai_text(response_data)

    def _post_json(
        self,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Send a JSON HTTP request using Python standard library only.

        Args:
            endpoint: Target URL.
            headers: HTTP headers.
            payload: JSON body.
            timeout_seconds: Request timeout in seconds.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: Raised when the remote endpoint returns an error.
        """
        http_request = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Request failed: {exc}") from exc

    def _extract_openai_text(self, response_data: dict[str, Any]) -> str:
        """Extract plain text from an OpenAI-compatible response.

        Args:
            response_data: Parsed JSON response body.

        Returns:
            Generated text content.
        """
        choices = response_data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        parts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)

        return "\n".join(part.strip() for part in parts if part.strip()).strip()
