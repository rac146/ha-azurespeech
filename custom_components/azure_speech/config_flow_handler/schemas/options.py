"""Schema definitions for options flow in Azure Speech config flow."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from custom_components.azure_speech.const import (
    CONF_AUDIO_FORMAT,
    CONF_LANGUAGE,
    CONF_PITCH,
    CONF_RATE,
    CONF_STYLE,
    CONF_STYLE_DEGREE,
    CONF_VOLUME,
    CONF_VOICE,
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_LANGUAGE,
    DEFAULT_PITCH,
    DEFAULT_RATE,
    DEFAULT_STYLE,
    DEFAULT_STYLE_DEGREE,
    DEFAULT_VOLUME,
    DEFAULT_VOICE,
)
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)


def get_options_schema(
    voices_list: list[dict[str, Any]],
    current_options: dict[str, Any],
) -> vol.Schema:
    """Build the options schema with dynamic voices dropdown."""
    voice_options: list[SelectOptionDict] = []
    discovered_styles: set[str] = set()

    if voices_list:
        for v in voices_list:
            short_name = v.get("ShortName", "")
            display_name = f"{short_name} ({v.get('Gender', '')}, {v.get('Locale', '')})"
            voice_options.append(SelectOptionDict(value=short_name, label=display_name))

            style_list = v.get("StyleList", [])
            if isinstance(style_list, list):
                discovered_styles.update(
                    style.strip()
                    for style in style_list
                    if isinstance(style, str) and style.strip()
                )

    selected_voice = current_options.get(CONF_VOICE, DEFAULT_VOICE)
    selected_style = current_options.get(CONF_STYLE, DEFAULT_STYLE)

    style_options = [
        SelectOptionDict(
            value=DEFAULT_STYLE,
            label="default",
        )
    ]
    style_options.extend(
        SelectOptionDict(
            value=style_name,
            label=style_name,
        )
        for style_name in sorted(discovered_styles)
    )

    if selected_style and selected_style not in {option["value"] for option in style_options}:
        style_options.append(SelectOptionDict(value=selected_style, label=selected_style))

    return vol.Schema(
        {
            vol.Optional(
                CONF_VOICE,
                default=selected_voice,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=voice_options or [selected_voice],
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_LANGUAGE,
                default=current_options.get(CONF_LANGUAGE, DEFAULT_LANGUAGE),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                CONF_STYLE,
                default=selected_style,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=style_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_STYLE_DEGREE,
                default=current_options.get(CONF_STYLE_DEGREE, DEFAULT_STYLE_DEGREE),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                CONF_PITCH,
                default=current_options.get(CONF_PITCH, DEFAULT_PITCH),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                CONF_RATE,
                default=current_options.get(CONF_RATE, DEFAULT_RATE),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                CONF_VOLUME,
                default=current_options.get(CONF_VOLUME, DEFAULT_VOLUME),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                CONF_AUDIO_FORMAT,
                default=current_options.get(CONF_AUDIO_FORMAT, DEFAULT_AUDIO_FORMAT),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=["mp3", "wav", "ogg"],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        }
    )
