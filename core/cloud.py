"""
Tiny clients for the cloud AI APIs whose keys are in the project's .env file.

  GEMINI_API_KEY -> Google Gemini (understands images: captions, OCR, zero-shot labels)
  GROQ_API_KEY   -> Groq (fast text LLMs: translation, rewriting)

Keys are read from .env only, sent only in request HEADERS, and never printed, logged or
shown in the app. Error messages never contain a key.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# 3.5-flash answered reliably when this was written; the newer ones timed out / were overloaded
GEMINI_MODELS = ("gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash")
GROQ_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")


@lru_cache(maxsize=1)
def _env() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    values = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def has_key(name: str) -> bool:
    return bool(_env().get(name))


def _key(name: str) -> str:
    value = _env().get(name)
    if not value:
        raise RuntimeError(f"{name} is missing - add it to the .env file in the project folder.")
    return value


def _post(url: str, payload: dict, headers: dict, timeout: int = 60, attempts: int = 3) -> dict:
    data = json.dumps(payload).encode()
    # a normal User-Agent: Groq's firewall blocks Python's default one
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "dip-studio/1.0", **headers})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503) and attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))                   # busy / rate limited: wait and retry
                continue
            detail = err.read().decode(errors="replace")[:300]
            raise RuntimeError(f"API error {err.code}: {detail}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
            if attempt < attempts - 1:                          # flaky connection: retry
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Could not reach the API (internet?): {getattr(err, 'reason', err)}") from None
    raise RuntimeError("API request failed")


def image_to_jpeg_b64(image: np.ndarray, max_side: int = 1024) -> str:
    h, w = image.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        image = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def gemini(prompt: str, image: np.ndarray | None = None, model: str = GEMINI_MODELS[0],
           temperature: float = 0.2) -> str:
    """Ask Gemini a question, optionally about an image (BGR numpy array)."""
    parts: list[dict] = [{"text": prompt}]
    if image is not None:
        parts.insert(0, {"inline_data": {"mime_type": "image/jpeg", "data": image_to_jpeg_b64(image)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": temperature}}
    data = _post(GEMINI_URL.format(model=model), payload, {"x-goog-api-key": _key("GEMINI_API_KEY")})
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini returned no text (blocked or empty): {str(data)[:200]}") from None


def gemini_image(prompt: str, model: str = "gemini-3.1-flash-image", image: np.ndarray | None = None) -> np.ndarray:
    """Generate (or edit, if `image` is given) a picture with a Gemini image model -> BGR array."""
    parts: list[dict] = [{"text": prompt}]
    if image is not None:
        parts.insert(0, {"inline_data": {"mime_type": "image/jpeg", "data": image_to_jpeg_b64(image)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}}
    data = _post(GEMINI_URL.format(model=model), payload, {"x-goog-api-key": _key("GEMINI_API_KEY")}, timeout=120)
    for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        blob = part.get("inline_data") or part.get("inlineData")
        if blob:
            return cv2.imdecode(np.frombuffer(base64.b64decode(blob["data"]), np.uint8), cv2.IMREAD_COLOR)
    raise RuntimeError(f"No image in the answer: {str(data)[:200]}")


def groq(prompt: str, model: str = GROQ_MODELS[0], temperature: float = 0.3) -> str:
    """Ask a Groq text model (OpenAI-compatible chat API)."""
    payload = {"model": model, "temperature": temperature,
               "messages": [{"role": "user", "content": prompt}]}
    data = _post(GROQ_URL, payload, {"Authorization": f"Bearer {_key('GROQ_API_KEY')}"})
    return data["choices"][0]["message"]["content"].strip()
