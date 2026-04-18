"""
Multi-modal message builder for LangChain and CrewAI.
Builds content suitable for HumanMessage (text + image_url parts) or CrewAI task descriptions.
"""

import base64
from typing import List, Optional, Union, Any

from utils.media_storage import MediaItem, ALLOWED_IMAGE_TYPES
from utils.logging_utils import get_logger

logger = get_logger("message_builder")

# Max size to fetch for inline base64 (avoid huge payloads). 10MB.
_MAX_INLINE_IMAGE_BYTES = 10 * 1024 * 1024
_FETCH_TIMEOUT_SECONDS = 30


def build_content(
    query: str,
    media_items: Optional[List[Any]] = None,
    use_inline_images: bool = True,
) -> Union[str, List[dict]]:
    """
    Build LangChain-compatible content for HumanMessage.

    - If no media_items or empty: returns the query string (text-only).
    - If media_items: returns a list of content parts:
      [{"type": "text", "text": query}, {"type": "image_url", "image_url": {"url": ...}}, ...]
    - By default (use_inline_images=True), images are sent as data:image/...;base64,... so the
      LLM provider does not need to fetch URLs (avoids provider-side timeouts on GCS signed URLs).
    - If use_inline_images=False, uses signed_url directly (provider will fetch; may timeout).

    Args:
        query: User text query.
        media_items: Optional list of MediaItem (or dicts with signed_url/media_type).
        use_inline_images: If True, fetch image from URL in this process and send as base64.

    Returns:
        Either a string (query only) or a list of content part dicts for multi-modal input.
    """
    if not media_items:
        return query

    parts: List[dict] = []
    if query and query.strip():
        parts.append({"type": "text", "text": query.strip()})

    for item in media_items:
        url = _get_url_from_media_item(item)
        media_type = _get_media_type_from_media_item(item)
        if not url:
            continue
        if media_type in ALLOWED_IMAGE_TYPES:
            image_url = url
            if use_inline_images:
                data_url = _url_to_data_url(url, media_type)
                if data_url:
                    image_url = data_url
                else:
                    logger.warning(
                        "[multimodal] build_content: could not fetch image for inline use, falling back to URL"
                    )
            parts.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })
            logger.debug(f"Added image part: {media_type}")
        # Video: OpenAI does not support native video; skip until Layer 8 (frame extraction)
        else:
            logger.debug(f"Skipping non-image media type for content: {media_type}")

    if not parts:
        return query or ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    return parts


def _url_to_data_url(url: str, media_type: str) -> Optional[str]:
    """
    Fetch image from URL (from this process, e.g. backend) and return a data URL.
    Avoids LLM provider having to fetch the URL (which can timeout from their network).
    """
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "AI-Agent-Framework/1.0"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
            content = resp.read()
            if len(content) > _MAX_INLINE_IMAGE_BYTES:
                logger.warning(
                    "[multimodal] Image size %s exceeds max inline size %s, using URL",
                    len(content),
                    _MAX_INLINE_IMAGE_BYTES,
                )
                return None
            b64 = base64.b64encode(content).decode("ascii")
            return f"data:{media_type};base64,{b64}"
    except Exception as e:
        logger.warning("[multimodal] Failed to fetch image for inline use: %s", e)
        return None


def build_crew_task_description(
    query: str,
    media_items: Optional[List[Any]] = None,
    use_inline_images: bool = False,
) -> str:
    """
    Build task description string for CrewAI Task(description=...).
    When media_items is present, appends image URLs (or data URLs when use_inline_images=True).

    Args:
        query: User text query.
        media_items: Optional list of MediaItem (or dicts with signed_url/media_type).
        use_inline_images: If True, fetch images and embed as data:image/...;base64,....
            Default False for Crew: embedding base64 in the task description blows up context
            length (millions of chars) and triggers "context length exceeded" / summarization.
            Use short URLs here; prefer LangGraph for reliable image handling (it uses
            build_content with inline images in the message parts).

    Returns:
        Single string: query plus optional "Attached image(s): <urls or data URLs>".
    """
    logger.info(
        "[multimodal] build_crew_task_description: query_len=%s media_items count=%s",
        len(query or ""), len(media_items) if media_items else 0,
    )
    if not media_items:
        return query

    urls: List[str] = []
    for i, item in enumerate(media_items):
        url = _get_url_from_media_item(item)
        media_type = _get_media_type_from_media_item(item)
        logger.info(
            "[multimodal] build_crew_task_description: item[%s] url_present=%s media_type=%s allowed=%s",
            i, bool(url), media_type, media_type in ALLOWED_IMAGE_TYPES,
        )
        if url and media_type in ALLOWED_IMAGE_TYPES:
            if use_inline_images:
                data_url = _url_to_data_url(url, media_type)
                if data_url:
                    url = data_url
                else:
                    logger.warning(
                        "[multimodal] build_crew_task_description: could not fetch image for inline use, using URL"
                    )
            urls.append(url)

    if not urls:
        logger.warning("[multimodal] build_crew_task_description: no valid image URLs collected, returning query only")
        return query

    suffix = "Attached image(s) (analyze when responding): " + " ".join(urls)
    out = f"{query}\n\n{suffix}" if query.strip() else suffix
    logger.info("[multimodal] build_crew_task_description: output len=%s url_count=%s", len(out), len(urls))
    return out


def _get_url_from_media_item(item: Any) -> Optional[str]:
    """Extract URL from MediaItem or dict."""
    if hasattr(item, "signed_url"):
        return getattr(item, "signed_url", None)
    if isinstance(item, dict):
        return item.get("signed_url") or item.get("url")
    return None


def _get_media_type_from_media_item(item: Any) -> str:
    """Extract media_type from MediaItem or dict."""
    if hasattr(item, "media_type"):
        return getattr(item, "media_type", "") or ""
    if isinstance(item, dict):
        return item.get("media_type", "") or ""
    return ""


def media_items_from_attachment_metadata(
    storage: Any,
    attachments: Optional[List[dict]] = None,
) -> List[MediaItem]:
    """
    Convert attachment metadata (from Interaction.user_attachments) into MediaItems
    with signed URLs, for use when building chat history messages.

    Args:
        storage: Storage instance with get_signed_url(cloud_path) method (e.g. GCSMediaStorage).
        attachments: List of dicts with cloud_path, media_type, media_id (optional).

    Returns:
        List of MediaItem with signed_url set; empty if storage is None or no valid attachments.
    """
    if not storage or not attachments:
        return []
    items: List[MediaItem] = []
    for a in attachments:
        if not isinstance(a, dict):
            continue
        cloud_path = a.get("cloud_path")
        if not cloud_path:
            continue
        try:
            signed_url = storage.get_signed_url(cloud_path)
        except Exception as e:
            logger.debug(f"Could not get signed URL for {cloud_path}: {e}")
            continue
        items.append(
            MediaItem(
                media_id=a.get("media_id", ""),
                filename=a.get("filename", cloud_path.split("/")[-1] if "/" in cloud_path else "attachment"),
                media_type=a.get("media_type", "application/octet-stream"),
                cloud_path=cloud_path,
                file_size=a.get("file_size", 0),
                signed_url=signed_url,
            )
        )
    return items
