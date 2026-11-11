"""Service-specific speech recognition settings."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from custom_components.azure_speech.const import (
    CONF_LANGUAGE,
    CONF_PROFANITY_MODE,
    CONF_STT_CANDIDATE_LANGUAGES,
    CONF_STT_ENDPOINT,
    CONF_STT_LANGUAGE,
    CONF_STT_LANGUAGE_MODE,
    CONF_STT_PHRASE_LIST,
    CONF_STT_PHRASE_LIST_WEIGHT,
    CONF_STT_POST_REFINEMENT,
    CONF_STT_PROMPT,
    CONF_STT_SERVICE,
    CONF_STT_TARGET_LANGUAGE,
    CONF_STT_TASK,
    CONF_STT_TEXT_FORMAT,
    CONF_STT_TIMEOUT,
    CONF_STT_TRANSCRIPT_STYLE,
    DEFAULT_LANGUAGE,
    DEFAULT_PROFANITY_MODE,
    DEFAULT_STT_LANGUAGE_MODE,
    DEFAULT_STT_POST_REFINEMENT,
    DEFAULT_STT_SERVICE,
    DEFAULT_STT_TIMEOUT,
    STT_SERVICE_FAST,
    STT_SERVICE_LLM,
    STT_SERVICE_MAI,
    STT_SERVICE_REALTIME,
    STT_SERVICE_SHORT_AUDIO,
)
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)


def _select(options: list[str], translation_key: str) -> SelectSelector:
    """Build a translated dropdown."""
    return SelectSelector(
        SelectSelectorConfig(options=options, mode=SelectSelectorMode.DROPDOWN, translation_key=translation_key)
    )


def get_stt_service_schema(current_options: dict[str, Any]) -> vol.Schema:
    """Select the recognition service before showing its supported controls."""
    return vol.Schema(
        {
            vol.Required(CONF_STT_SERVICE, default=current_options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE)): _select(
                [STT_SERVICE_SHORT_AUDIO, STT_SERVICE_REALTIME, STT_SERVICE_FAST, STT_SERVICE_LLM, STT_SERVICE_MAI],
                "stt_service",
            ),
        }
    )


def get_stt_options_schema(current_options: dict[str, Any]) -> vol.Schema:
    """Build a schema containing only options supported by the selected service."""
    service = current_options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE)
    language_modes = ["pipeline", "fixed"]
    if service not in (STT_SERVICE_SHORT_AUDIO, STT_SERVICE_REALTIME):
        language_modes.append("automatic")

    mode = current_options.get(CONF_STT_LANGUAGE_MODE, DEFAULT_STT_LANGUAGE_MODE)
    if mode not in language_modes:
        mode = DEFAULT_STT_LANGUAGE_MODE

    fields: dict[Any, Any] = {
        vol.Optional(CONF_STT_LANGUAGE_MODE, default=mode): _select(language_modes, "stt_language_mode"),
        vol.Optional(
            CONF_STT_LANGUAGE,
            default=current_options.get(CONF_STT_LANGUAGE, current_options.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)),
        ): TextSelector(TextSelectorConfig()),
        vol.Optional(CONF_PROFANITY_MODE, default=current_options.get(CONF_PROFANITY_MODE, DEFAULT_PROFANITY_MODE)): _select(
            ["masked", "removed", "raw"], "profanity_mode"
        ),
        vol.Optional(CONF_STT_ENDPOINT, default=current_options.get(CONF_STT_ENDPOINT, "")): TextSelector(
            TextSelectorConfig()
        ),
        vol.Optional(CONF_STT_TIMEOUT, default=current_options.get(CONF_STT_TIMEOUT, DEFAULT_STT_TIMEOUT)): NumberSelector(
            NumberSelectorConfig(min=5, max=120, step=1, mode=NumberSelectorMode.BOX, unit_of_measurement="s")
        ),
    }
    if service in (STT_SERVICE_SHORT_AUDIO, STT_SERVICE_REALTIME):
        fields[vol.Optional(CONF_STT_TEXT_FORMAT, default=current_options.get(CONF_STT_TEXT_FORMAT, "display"))] = _select(
            ["display", "lexical"], "stt_text_format"
        )
        if service == STT_SERVICE_SHORT_AUDIO:
            return vol.Schema(fields)
        fields[
            vol.Optional(
                CONF_STT_POST_REFINEMENT,
                default=current_options.get(CONF_STT_POST_REFINEMENT, DEFAULT_STT_POST_REFINEMENT),
            )
        ] = BooleanSelector()

    fields[vol.Optional(CONF_STT_PHRASE_LIST, default=current_options.get(CONF_STT_PHRASE_LIST, ""))] = TextSelector(
        TextSelectorConfig(multiline=True)
    )
    if service in (STT_SERVICE_REALTIME, STT_SERVICE_FAST, STT_SERVICE_LLM):
        fields[
            vol.Optional(CONF_STT_PHRASE_LIST_WEIGHT, default=current_options.get(CONF_STT_PHRASE_LIST_WEIGHT, 1.0))
        ] = NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.1, mode=NumberSelectorMode.BOX))

    if service == STT_SERVICE_FAST:
        fields[
            vol.Optional(CONF_STT_CANDIDATE_LANGUAGES, default=current_options.get(CONF_STT_CANDIDATE_LANGUAGES, ""))
        ] = TextSelector(TextSelectorConfig(multiline=True))
    elif service == STT_SERVICE_LLM:
        fields[vol.Optional(CONF_STT_TASK, default=current_options.get(CONF_STT_TASK, "transcribe"))] = _select(
            ["transcribe", "translate"], "stt_task"
        )
        fields[
            vol.Optional(CONF_STT_TARGET_LANGUAGE, default=current_options.get(CONF_STT_TARGET_LANGUAGE, ""))
        ] = TextSelector(TextSelectorConfig())
        fields[vol.Optional(CONF_STT_PROMPT, default=current_options.get(CONF_STT_PROMPT, ""))] = TextSelector(
            TextSelectorConfig(multiline=True)
        )
    elif service == STT_SERVICE_MAI:
        fields[
            vol.Optional(CONF_STT_TRANSCRIPT_STYLE, default=current_options.get(CONF_STT_TRANSCRIPT_STYLE, "verbatim"))
        ] = _select(["verbatim", "clean"], "stt_transcript_style")

    return vol.Schema(fields)
