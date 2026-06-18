"""OpenAI vision description for a single photo.

One chat call per image returns a strict-JSON object with a prose description
(for semantic embedding) plus structured attributes tuned to the gallery's
filter axes. khora's own ``acompletion`` is text-only, so we call the OpenAI
SDK directly with an image message (same pattern as the repo's
``examples/10_core_apis/07_image_ingestion.py``).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from PIL import Image, ImageOps

VISION_MODEL = os.environ.get("PHOTO_VISION_MODEL", "gpt-4o-mini")
# Longest edge sent to the vision model. The original can be tens of megapixels;
# sending it whole wastes memory and tokens, and llama.cpp ignores the OpenAI
# `detail` hint, so we downscale ourselves.
VISION_MAX_DIM = int(os.environ.get("PHOTO_VISION_MAX_DIM", "768"))

_SYSTEM = (
    "You describe photographs for a searchable gallery. Return STRICT JSON with keys:\n"
    '  "description": 2-4 vivid sentences capturing subject, setting, mood (for search).\n'
    '  "location": the place or setting as a short phrase ("beach", "kitchen", "Paris street") or "".\n'
    '  "objects": array of notable things/structures (lowercase short nouns).\n'
    '  "animals": array of animals present, [] if none.\n'
    '  "scene": the activity or scene type ("wedding", "hiking", "sunset") or "".\n'
    '  "tags": array of 3-8 lowercase keywords for filtering.\n'
    "Describe what you actually see. NEVER output placeholder or bracketed text "
    "like [Description] or [Location] — fill every field from the image itself.\n"
    "Output ONLY the JSON object."
)

# Small vision models sometimes echo the field-name placeholders ("[Description]
# [Location] [Objects]") instead of describing the image — strip those so the
# garbage never reaches the embedding/search index.
_PLACEHOLDER_RE = re.compile(r"\[\s*(?:description|location|objects?|animals?|scenes?|tags?)\s*\]", re.IGNORECASE)


def _clean_text(text: str) -> str:
    """Remove echoed [Field] placeholders and tidy the leftover punctuation."""
    cleaned = _PLACEHOLDER_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip(" .,:;-")

def _as_list(v: Any) -> list[str]:
    """Coerce an array field to a clean list. Small models sometimes return a
    comma-joined string instead of a JSON array — split it, rather than letting
    a downstream ``for x in <string>`` iterate it character by character."""
    if isinstance(v, str):
        parts = v.split(",")
    elif isinstance(v, list):
        parts = v
    else:
        return []
    return [str(x).strip() for x in parts if str(x).strip()]


# A JSON schema (not just json_object): llama.cpp turns this into a grammar and
# *constrains* generation to valid, complete JSON. Plain json_object isn't
# grammar-enforced, so small vision models otherwise emit malformed or runaway
# output that fails json.loads and the photo fails to scan.
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "description": {"type": "string"},
        "location": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "animals": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "location", "objects", "animals", "scene", "tags"],
}
_RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "photo", "schema": _SCHEMA}}


def _loads(content: str) -> dict[str, Any]:
    """Parse the model's JSON, tolerating a stray ```fence``` or surrounding prose."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                pass
        # Runaway/truncated output (model looped past max_tokens): salvage the
        # description so the photo still scans instead of failing outright.
        m = re.search(r'"description"\s*:\s*"(.*?)(?:"\s*[,}]|$)', text, re.DOTALL)
        if m:
            return {"description": m.group(1).strip()[:600]}
        raise


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()  # reads OPENAI_API_KEY from env
    return _client


def _encode_image(path: Path) -> str:
    """Downscale (respecting EXIF orientation) and re-encode to a base64 JPEG."""
    img = ImageOps.exif_transpose(Image.open(path))
    img.thumbnail((VISION_MAX_DIM, VISION_MAX_DIM))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


async def describe_image(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (attributes, usage) where usage = {model, input, output} token counts."""
    mime = "image/jpeg"
    b64 = _encode_image(path)
    resp = await _get_client().chat.completions.create(
        model=VISION_MODEL,
        response_format=_RESPONSE_FORMAT,
        max_tokens=700,  # cap runaway generation on small models
        temperature=0.2,  # low temp + repeat penalty curb degenerate loops
        extra_body={"repeat_penalty": 1.3},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this photo as specified."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
                    },
                ],
            },
        ],
    )
    obj = _loads(resp.choices[0].message.content or "{}")
    u = resp.usage
    usage = {
        "model": VISION_MODEL,
        "input": getattr(u, "prompt_tokens", 0) or 0,
        "output": getattr(u, "completion_tokens", 0) or 0,
    }
    # Clean each field of echoed [Field] placeholders; drop list items that are
    # nothing but a placeholder.
    drop = lambda xs: [x for x in xs if x and not _PLACEHOLDER_RE.fullmatch(x.strip())]
    description = _clean_text(str(obj.get("description", "")))
    location = _clean_text(str(obj.get("location", "")))
    scene = _clean_text(str(obj.get("scene", "")))
    objects = drop(_as_list(obj.get("objects")))
    animals = drop(_as_list(obj.get("animals")))
    tags = [t.lower() for t in drop(_as_list(obj.get("tags")))]

    # If the model gave only placeholders/empty prose, synthesize a description
    # from the attributes it did extract so the photo still has real search
    # signal instead of polluting the index with "[Description] [Location] ...".
    if not description:
        parts = [p for p in ([scene, location] + objects + animals + tags) if p]
        description = ", ".join(dict.fromkeys(parts))  # de-duped, order preserved

    data = {
        "description": description,
        "location": location,
        "objects": objects,
        "animals": animals,
        "scene": scene,
        "tags": tags,
    }
    return data, usage
