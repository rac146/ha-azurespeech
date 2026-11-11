"""Validate STT settings and map them to Azure transcription request fields.

The fast transcription endpoint uses API version 2025-10-15 for phrase lists
and enhanced mode. Post-stream refinement is a Speech SDK feature and is not
part of this REST API. See Microsoft's fast-transcription-create, llm-speech,
mai-transcribe, and improve-accuracy-phrase-list documentation.
"""

from __future__ import annotations

import re
from typing import Any

from yarl import URL

from custom_components.azure_speech.const import (
    CONF_PROFANITY_MODE,
    CONF_STT_CANDIDATE_LANGUAGES,
    CONF_STT_ENDPOINT,
    CONF_STT_LANGUAGE,
    CONF_STT_LANGUAGE_MODE,
    CONF_STT_PHRASE_LIST,
    CONF_STT_PHRASE_LIST_WEIGHT,
    CONF_STT_PROMPT,
    CONF_STT_SERVICE,
    CONF_STT_TARGET_LANGUAGE,
    CONF_STT_TASK,
    CONF_STT_TEXT_FORMAT,
    CONF_STT_TIMEOUT,
    CONF_STT_TRANSCRIPT_STYLE,
    DEFAULT_PROFANITY_MODE,
    DEFAULT_STT_LANGUAGE_MODE,
    DEFAULT_STT_SERVICE,
    DEFAULT_STT_TIMEOUT,
    STT_MAX_PHRASES,
    STT_MAX_PROMPT_LENGTH,
    STT_SERVICE_FAST,
    STT_SERVICE_LLM,
    STT_SERVICE_MAI,
    STT_SERVICE_REALTIME,
    STT_SERVICE_SHORT_AUDIO,
)


def split_stt_list(value: str, *, locales: bool = False) -> list[str]:
    """Read newline/semicolon separated phrases, preserving order without duplicates."""
    separators = r"[\n;,]" if locales else r"[\n;]"
    return list(dict.fromkeys(item.strip() for item in re.split(separators, value) if item.strip()))


def validate_stt_options(options: dict[str, Any]) -> dict[str, str]:
    """Return field errors without making a billable recognition request."""
    errors: dict[str, str] = {}
    service = options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE)
    if service not in (
        STT_SERVICE_SHORT_AUDIO,
        STT_SERVICE_REALTIME,
        STT_SERVICE_FAST,
        STT_SERVICE_LLM,
        STT_SERVICE_MAI,
    ):
        errors[CONF_STT_SERVICE] = "invalid_stt_service"

    language_mode = options.get(CONF_STT_LANGUAGE_MODE, DEFAULT_STT_LANGUAGE_MODE)
    allowed_modes = (
        ("pipeline", "fixed")
        if service in (STT_SERVICE_SHORT_AUDIO, STT_SERVICE_REALTIME)
        else ("pipeline", "fixed", "automatic")
    )
    if language_mode not in allowed_modes:
        errors[CONF_STT_LANGUAGE_MODE] = "invalid_language_mode"
    if language_mode == "fixed" and not re.fullmatch(r"[a-z]{2,3}-[A-Za-z0-9-]{2,15}", options.get(CONF_STT_LANGUAGE, "")):
        errors[CONF_STT_LANGUAGE] = "invalid_stt_language"

    endpoint = options.get(CONF_STT_ENDPOINT, "")
    if endpoint:
        try:
            url = URL(endpoint)
            if url.scheme != "https" or not url.host or url.user or url.query_string or url.fragment or url.path != "/":
                errors[CONF_STT_ENDPOINT] = "invalid_stt_endpoint"
        except ValueError:
            errors[CONF_STT_ENDPOINT] = "invalid_stt_endpoint"

    timeout = options.get(CONF_STT_TIMEOUT, DEFAULT_STT_TIMEOUT)
    if not isinstance(timeout, (int, float)) or not 5 <= timeout <= 120:
        errors[CONF_STT_TIMEOUT] = "invalid_stt_timeout"
    if options.get(CONF_PROFANITY_MODE, DEFAULT_PROFANITY_MODE) not in ("masked", "removed", "raw"):
        errors[CONF_PROFANITY_MODE] = "invalid_profanity_mode"

    if service in (STT_SERVICE_SHORT_AUDIO, STT_SERVICE_REALTIME):
        if options.get(CONF_STT_TEXT_FORMAT, "display") not in ("display", "lexical"):
            errors[CONF_STT_TEXT_FORMAT] = "invalid_text_format"
        if service == STT_SERVICE_SHORT_AUDIO:
            return errors

    phrases = split_stt_list(options.get(CONF_STT_PHRASE_LIST, ""))
    if len(phrases) > STT_MAX_PHRASES:
        errors[CONF_STT_PHRASE_LIST] = "too_many_phrases"
    if service in (STT_SERVICE_REALTIME, STT_SERVICE_FAST, STT_SERVICE_LLM):
        weight = options.get(CONF_STT_PHRASE_LIST_WEIGHT, 1.0)
        if not isinstance(weight, (int, float)) or not 0 <= weight <= 2:
            errors[CONF_STT_PHRASE_LIST_WEIGHT] = "invalid_phrase_weight"
    if service == STT_SERVICE_FAST:
        candidates = split_stt_list(options.get(CONF_STT_CANDIDATE_LANGUAGES, ""), locales=True)
        if any(not re.fullmatch(r"[a-z]{2,3}-[A-Za-z0-9-]{2,15}", locale) for locale in candidates):
            errors[CONF_STT_CANDIDATE_LANGUAGES] = "invalid_stt_language"
    elif service == STT_SERVICE_LLM:
        task = options.get(CONF_STT_TASK, "transcribe")
        if task not in ("transcribe", "translate"):
            errors[CONF_STT_TASK] = "invalid_stt_task"
        if task == "translate" and not re.fullmatch(r"[a-z]{2,3}(?:-[A-Z][a-z]{3})?", options.get(CONF_STT_TARGET_LANGUAGE, "")):
            errors[CONF_STT_TARGET_LANGUAGE] = "invalid_target_language"
        if len(options.get(CONF_STT_PROMPT, "")) > STT_MAX_PROMPT_LENGTH:
            errors[CONF_STT_PROMPT] = "prompt_too_long"
    elif service == STT_SERVICE_MAI and options.get(CONF_STT_TRANSCRIPT_STYLE, "verbatim") not in ("verbatim", "clean"):
        errors[CONF_STT_TRANSCRIPT_STYLE] = "invalid_transcript_style"

    return errors


def build_transcription_definition(options: dict[str, Any], language: str, profanity_mode: str) -> dict[str, Any]:
    """Build a service-specific definition, omitting unsupported and inactive settings."""
    service = options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE)
    definition: dict[str, Any] = {
        "profanityFilterMode": {"masked": "Masked", "removed": "Removed", "raw": "None"}[profanity_mode],
    }
    mode = options.get(CONF_STT_LANGUAGE_MODE, DEFAULT_STT_LANGUAGE_MODE)
    if mode != "automatic":
        definition["locales"] = [language.split("-", maxsplit=1)[0] if service == STT_SERVICE_MAI else language]
    elif service == STT_SERVICE_FAST:
        candidates = split_stt_list(options.get(CONF_STT_CANDIDATE_LANGUAGES, ""), locales=True)
        if candidates:
            definition["locales"] = candidates

    phrases = split_stt_list(options.get(CONF_STT_PHRASE_LIST, ""))
    if phrases:
        definition["phraseList"] = {"phrases": phrases}
        if service in (STT_SERVICE_FAST, STT_SERVICE_LLM):
            definition["phraseList"]["biasingWeight"] = options.get(CONF_STT_PHRASE_LIST_WEIGHT, 1.0)

    if service == STT_SERVICE_LLM:
        task = options.get(CONF_STT_TASK, "transcribe")
        enhanced: dict[str, Any] = {"enabled": True, "task": task}
        if task == "translate":
            enhanced["targetLanguage"] = options[CONF_STT_TARGET_LANGUAGE]
        if prompt := options.get(CONF_STT_PROMPT, "").strip():
            enhanced["prompt"] = [prompt]
        definition["enhancedMode"] = enhanced
    elif service == STT_SERVICE_MAI:
        definition["enhancedMode"] = {
            "enabled": True,
            "model": "MAI-Transcribe-2",
            "modelOptions": {"transcribeStyle": options.get(CONF_STT_TRANSCRIPT_STYLE, "verbatim")},
        }

    return definition
