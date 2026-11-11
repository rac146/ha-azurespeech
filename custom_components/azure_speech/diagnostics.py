"""Diagnostics support for azure_speech custom integration."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.redact import async_redact_data

from .const import CONF_API_KEY, CONF_ENDPOINT, CONF_STT_ENDPOINT, CONF_STT_PHRASE_LIST, CONF_STT_PROMPT
from .data import AzureSpeechConfigEntry

TO_REDACT = {
    CONF_API_KEY,
    CONF_ENDPOINT,
    CONF_STT_ENDPOINT,
    CONF_STT_PHRASE_LIST,
    CONF_STT_PROMPT,
    "api_key",
    "subscription_key",
    "password",
    "token",
}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,
    entry: AzureSpeechConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    coordinator = entry.runtime_data.coordinator

    return {
        "entry_data": async_redact_data(dict(entry.data), TO_REDACT),
        "entry_options": async_redact_data(dict(entry.options), TO_REDACT),
        "coordinator_data": {
            "voices_count": coordinator.data.get("voices_count", 0),
            "status": coordinator.data.get("status", "unknown"),
        },
    }
