import asyncio
import base64
import json
import mimetypes
import os
import re
from typing import Any
from urllib import error, request

from astrbot.api import logger


class PluginLLMRouter:
    """按角色路由插件 LLM 调用，不改 AstrBot 核心。

    路由支持两种模式：
    1. ``astrbot_default``：全部继续走 AstrBot 当前提供方。
    2. ``plugin_router``：按角色调用插件内配置的 OpenAI 兼容接口。

    Args:
        context: AstrBot 插件上下文。
        settings: ``config.yaml`` 里的 ``llm_settings`` 配置段。
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
        """为指定角色生成文本。

        Args:
            role: 逻辑角色名，例如 ``chat`` 或 ``analysis``。
            prompt: 用户输入提示词。
            system_prompt: 系统提示词。
            context_messages: 可选的 OpenAI 消息格式上下文。
            image_urls: 可选图片 URL 列表，用于视觉模型。

        Returns:
            生成的文本内容。
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
                logger.error(f"[LLM 路由] 角色 {role} 的插件路由调用失败：{exc}")
                if not self.fallback_to_default:
                    raise
        return await self._call_astrbot_default(prompt, system_prompt)

    async def _call_astrbot_default(
        self,
        prompt: str,
        system_prompt: str,
    ) -> str:
        """调用当前 AstrBot provider。

        Args:
            prompt: 用户输入提示词。
            system_prompt: 系统提示词。

        Returns:
            生成的文本内容。
        """
        provider = self.context.get_using_provider()
        if not provider:
            raise RuntimeError("AstrBot 默认提供方不可用。")
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
        """调用插件配置里的 OpenAI 兼容接口。

        Args:
            role_config: 角色专用的 OpenAI 兼容配置。
            prompt: 用户输入提示词。
            system_prompt: 系统提示词。
            context_messages: 可选消息历史。
            image_urls: 可选图片 URL 列表，用于视觉请求。

        Returns:
            生成的文本内容。
        """
        if role_config.get("provider_type") != "openai_compatible":
            raise ValueError(
                f"不支持的 provider_type：{role_config.get('provider_type')}",
            )

        base_url = str(role_config.get("base_url", "")).strip()
        model = str(role_config.get("model", "")).strip()
        if not base_url or not model:
            raise ValueError("在 plugin_router 模式下必须提供 base_url 和 model。")

        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_messages)

        if image_urls:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for url in image_urls:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._normalize_image_reference(url)},
                    }
                )
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
        api_key = self._resolve_env_var(api_key)
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
        """只用标准库发送 JSON HTTP 请求。

        Args:
            endpoint: 目标 URL。
            headers: HTTP 请求头。
            payload: JSON 请求体。
            timeout_seconds: 请求超时时间，单位秒。

        Returns:
            解析后的 JSON 响应。

        Raises:
            RuntimeError: 当远端接口返回错误时抛出。
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
        """从 OpenAI 兼容响应中提取纯文本。

        Args:
            response_data: 已解析的 JSON 响应体。

        Returns:
            生成的文本内容。
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

    def _normalize_image_reference(self, value: str) -> str:
        """Normalize an image reference for OpenAI-compatible requests.

        Args:
            value: Image URL or local file path.

        Returns:
            A remote URL or a base64 data URL when the local file is readable.
        """
        image_ref = str(value).strip()
        if not image_ref:
            return image_ref
        if image_ref.startswith("data:image/"):
            return image_ref
        if image_ref.startswith(("http://", "https://")):
            return image_ref
        if image_ref.startswith("file://"):
            image_ref = image_ref[7:]
        if os.path.exists(image_ref) and os.path.isfile(image_ref):
            try:
                with open(image_ref, "rb") as handle:
                    data = handle.read()
                mime_type = mimetypes.guess_type(image_ref)[0] or "image/jpeg"
                encoded = base64.b64encode(data).decode("ascii")
                return f"data:{mime_type};base64,{encoded}"
            except Exception as exc:
                logger.debug(
                    f"[LLM Router] Failed to encode local image {image_ref}: {exc}"
                )
        return image_ref

    @staticmethod
    def _resolve_env_var(value: str) -> str:
        """Resolve ``${ENV_VAR}`` patterns in a config value.

        Args:
            value: Config value that may contain env var references.

        Returns:
            Resolved value with env vars substituted.
        """
        pattern = re.compile(r"\$\{([^}]+)\}")
        result = value
        for match in pattern.finditer(value):
            env_name = match.group(1)
            env_val = os.environ.get(env_name, "")
            result = result.replace(match.group(0), env_val)
        return result.strip()
