"""OpenAI-compatible vision-language backend for VLM attribute annotation.

``stage3_attribute.py`` historically ran a local Qwen3-VL model for attribute
annotation.  This module provides an equivalent network backend that talks to
any OpenAI-compatible ``/chat/completions`` endpoint (for example a GLM
``glm5.3flash`` service), so the heavy vision-language inference can move to an
external API while every downstream WordNet/ontology decision stays local.

Configuration (all optional except the endpoint and key):

    VLM_API_BASE             endpoint base, e.g. https://open.bigmodel.cn/api/paas/v4
    VLM_API_KEY              bearer token
    VLM_MODEL                model id, default "glm-5.3-flash" (note the hyphens)
    VLM_API_TIMEOUT          seconds per request (default 60)
    VLM_API_MAX_RETRIES      retries on 429/5xx/network errors (default 4)
    VLM_API_CONCURRENCY      parallel requests inside generate_raw_batch (default 4)
    VLM_API_IMAGE_MAX_EDGE   longest edge pixels sent to the API (default 1536)
    VLM_API_IMAGE_QUALITY    JPEG quality for uploaded images (default 85)
    VLM_API_TEMPERATURE      sampling temperature (default 0.0)
    VLM_API_REASONING_EFFORT "low"|"high"|"max"; GLM-5.3-Flash always reasons and
                             "low" keeps the reasoning tokens small (default "low").
                             Set to empty to omit the field for other providers.
    VLM_API_THINKING         "enabled"|"disabled" sent as {"thinking": {"type": ...}}.
                             GLM-5.3-Flash only accepts "enabled". Empty (default)
                             means the field is omitted.
    VLM_API_TOKEN_SCALE      multiplied into max_tokens for the API. Callers pass
                             small local-generation budgets that must also cover the
                             model's reasoning output (default 3).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import error as urllib_error
from urllib import request as urllib_request

from PIL import Image

DEFAULT_MODEL = "glm-5.3-flash"


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    """Never auto-follow redirects: the gateway turns POST into GET there."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib_error.HTTPError(
            req.full_url, code, msg, headers, fp
        )



def extract_json_object(text: str) -> dict:
    """Tolerant ``{...}`` JSON extraction mirroring stage3's local parser."""
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start:end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("VLM response is not a JSON object", text, 0)
    return parsed


class OpenAIChatBackend:
    """Vision chat backend speaking the OpenAI ``/chat/completions`` protocol."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        max_image_edge: int = 1536,
        jpeg_quality: int = 85,
        timeout: float = 60.0,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        concurrency: int = 4,
        temperature: float = 0.0,
        top_p: float | None = None,
        reasoning_effort: str | None = "low",
        thinking: str | None = None,
        token_scale: float = 3.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_image_edge = max(64, int(max_image_edge))
        self.jpeg_quality = int(jpeg_quality)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.concurrency = max(1, int(concurrency))
        self.temperature = float(temperature)
        self.top_p = float(top_p) if top_p is not None else None
        self.reasoning_effort = reasoning_effort or None
        self.thinking = thinking or None
        self.token_scale = max(1.0, float(token_scale))
        self._opener = urllib_request.build_opener(_NoRedirect())

    @property
    def name(self) -> str:
        return f"api:{self.model}"

    @property
    def chat_url(self) -> str:
        return f"{self.api_base}/chat/completions"

    def _encode_image(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        longest = max(image.size)
        if longest > self.max_image_edge:
            scale = self.max_image_edge / longest
            size = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            image = image.resize(size, Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _messages(self, images, prompt: str) -> list[dict]:
        if images is None:
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        image_list = images if isinstance(images, (list, tuple)) else [images]
        content = []
        for image in image_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(image)},
            })
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _post(self, messages: list[dict], max_new_tokens: int) -> str:
        budget = max(16, round(int(max_new_tokens) * self.token_scale))
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": budget,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib_request.Request(
            self.chat_url, data=body, headers=headers, method="POST"
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
            except urllib_error.HTTPError as exc:
                last_error = exc
                if exc.code in {301, 302, 303, 307, 308, 429, 500, 502, 503, 504}:
                    if attempt >= self.max_retries:
                        break
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    detail = ""
                raise RuntimeError(f"VLM API HTTP {exc.code}: {detail}") from exc
            except (urllib_error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff * (2 ** attempt))
                continue
            try:
                parsed = json.loads(response_body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"VLM API returned non-JSON: {response_body[:300]}"
                ) from exc
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                error = parsed["error"]
                raise RuntimeError(
                    f"VLM API error: {json.dumps(error, ensure_ascii=False)[:300]}"
                )
            try:
                content = parsed["choices"][0]["message"]["content"]
            except (KeyError, TypeError, IndexError) as exc:
                raise RuntimeError(
                    f"VLM API returned an unexpected payload: {response_body[:300]}"
                ) from exc
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            return str(content).strip()
        raise RuntimeError(f"VLM API request failed after retries: {last_error}")

    def generate_raw(self, images, prompt: str, max_new_tokens: int = 384) -> str:
        return self._post(self._messages(images, prompt), max_new_tokens)

    def generate_json(
        self, images, prompt: str, max_new_tokens: int = 384,
    ) -> tuple[dict, str]:
        raw = self.generate_raw(images, prompt, max_new_tokens=max_new_tokens)
        if not raw:
            raise RuntimeError(
                "VLM API 返回了空 content（可能推理 token 占满预算）；请调大 "
                "VLM_API_TOKEN_SCALE 或 VLM_API_REASONING_EFFORT=low"
            )
        return extract_json_object(raw), raw

    def generate_raw_batch(
        self,
        image_groups: list,
        prompts: str | list[str],
        max_new_tokens: int = 384,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(image_groups)
        if len(prompts) != len(image_groups):
            raise ValueError("Batch prompts and image groups must have equal length")
        results: list[str | None] = [None] * len(image_groups)
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_map = {
                executor.submit(
                    self.generate_raw, images, prompt,
                    max_new_tokens=max_new_tokens,
                ): index
                for index, (images, prompt) in enumerate(zip(image_groups, prompts))
            }
            for future in as_completed(future_map):
                index = future_map[future]
                results[index] = future.result()
        return results  # type: ignore[return-value]


def openai_backend_from_env() -> OpenAIChatBackend:
    """Build an OpenAIChatBackend from ``VLM_*`` environment variables."""
    api_base = os.environ.get("VLM_API_BASE", "").strip().rstrip("/")
    if not api_base:
        raise RuntimeError("VLM_API_BASE 未设置，无法使用 --vlm-backend api")
    api_key = os.environ.get("VLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("VLM_API_KEY 未设置，无法使用 --vlm-backend api")

    def env_int(name: str, default: int) -> int:
        value = os.environ.get(name, "").strip()
        try:
            return int(value) if value else default
        except ValueError:
            return default

    return OpenAIChatBackend(
        api_base,
        api_key,
        model=os.environ.get("VLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        max_image_edge=env_int("VLM_API_IMAGE_MAX_EDGE", 1536),
        jpeg_quality=env_int("VLM_API_IMAGE_QUALITY", 85),
        timeout=float(os.environ.get("VLM_API_TIMEOUT", "60")),
        max_retries=env_int("VLM_API_MAX_RETRIES", 4),
        concurrency=env_int("VLM_API_CONCURRENCY", 4),
        temperature=float(os.environ.get("VLM_API_TEMPERATURE", "0.0")),
        top_p=(
            float(os.environ["VLM_API_TOP_P"])
            if os.environ.get("VLM_API_TOP_P", "").strip()
            else None
        ),
        reasoning_effort=os.environ.get("VLM_API_REASONING_EFFORT", "low") or None,
        thinking=os.environ.get("VLM_API_THINKING", "") or None,
        token_scale=float(os.environ.get("VLM_API_TOKEN_SCALE", "3.0")),
    )
