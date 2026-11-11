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
    diagnostics: dict[str, Any] = {
        "entry_data": async_redact_data(dict(entry.data), TO_REDACT),
        "entry_options": async_redact_data(dict(entry.options), TO_REDACT),
        "request_trace_available": False,
        "last_stt_request": None,
        "last_stt_response": None,
        "coordinator_data": {
            "voices_count": 0,
            "status": "unknown",
        },
    }

    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is None:
        return diagnostics

    coordinator_data = runtime_data.coordinator.data or {}
    diagnostics["coordinator_data"] = {
        "voices_count": coordinator_data.get("voices_count", 0),
        "status": coordinator_data.get("status", "unknown"),
    }
    diagnostics["request_trace_available"] = hasattr(runtime_data.client, "last_stt_request")
    diagnostics["last_stt_request"] = async_redact_data(
        getattr(runtime_data.client, "last_stt_request", None), TO_REDACT
    )
    diagnostics["last_stt_response"] = async_redact_data(
        getattr(runtime_data.client, "last_stt_response", None), TO_REDACT
    )
    return diagnostics
