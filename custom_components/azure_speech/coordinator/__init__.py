"""Data update coordinator for Azure Speech integration."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from custom_components.azure_speech.api import (
    AzureSpeechApiClient,
    AzureSpeechApiClientAuthenticationError,
    AzureSpeechApiClientError,
    AzureSpeechApiClientRateLimitError,
)
from custom_components.azure_speech.const import DOMAIN, LOGGER
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

UPDATE_INTERVAL = timedelta(hours=12)


class AzureSpeechDataUpdateCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage fetching Azure Speech data (voices list and API health)."""

    config_entry: ConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        client: AzureSpeechApiClient,
    ) -> None:
        """Initialize the Azure Speech data update coordinator."""
        super().__init__(
            hass=hass,
            logger=LOGGER,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
        )
        self.client = client
        self.voices: list[dict[str, Any]] = []

    async def async_transcribe_audio(
        self,
        audio: bytes,
        language: str,
        profanity_mode: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Route recognition requests from the STT entity to the API client."""
        return await self.client.transcribe_stt_audio(
            wav_audio_bytes=audio,
            language=language,
            profanity_mode=profanity_mode,
            stt_options=options,
        )

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch updated voices list and verify API connectivity."""
        try:
            voices = await self.client.get_voices()
            self.voices = voices
            return {
                "voices": voices,
                "voices_count": len(voices),
                "status": "online",
            }
        except AzureSpeechApiClientAuthenticationError as err:
            raise ConfigEntryAuthFailed(f"Authentication failed for Azure Speech API: {err}") from err
        except AzureSpeechApiClientRateLimitError as err:
            raise UpdateFailed(f"Azure Speech API rate limit exceeded. Retry after {err.retry_after}s") from err
        except AzureSpeechApiClientError as err:
            raise UpdateFailed(f"Error communicating with Azure Speech API: {err}") from err
