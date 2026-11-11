"""Speech-to-text platform for azure_speech."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any

from custom_components.azure_speech.api import (
    AzureSpeechApiClientAuthenticationError,
    AzureSpeechApiClientCommunicationError,
    AzureSpeechApiClientError,
)
from custom_components.azure_speech.const import (
    CONF_LANGUAGE,
    CONF_PROFANITY_MODE,
    CONF_STT_LANGUAGE,
    CONF_STT_LANGUAGE_MODE,
    CONF_STT_SERVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_PROFANITY_MODE,
    DEFAULT_STT_LANGUAGE_MODE,
    DEFAULT_STT_SERVICE,
    LOGGER,
    STT_SERVICE_MAI,
)
from custom_components.azure_speech.data import AzureSpeechConfigEntry
from custom_components.azure_speech.entity import AzureSpeechEntity
from custom_components.azure_speech.entity_utils import wrap_pcm_in_wav
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityDescription
from homeassistant.helpers.entity_platform import AddEntitiesCallback


class AzureSpeechSTTEntity(SpeechToTextEntity, AzureSpeechEntity):
    """Azure Speech Speech-to-Text entity implementation."""

    def __init__(
        self,
        entry: AzureSpeechConfigEntry,
    ) -> None:
        """Initialize Azure Speech STT entity."""
        coordinator = entry.runtime_data.coordinator
        description = EntityDescription(
            key="stt",
            name="Azure Speech STT",
        )
        super().__init__(coordinator, description)
        self._entry = entry

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages for STT transcription."""
        return [
            "en-US",
            "en-GB",
            "en-AU",
            "en-CA",
            "de-DE",
            "fr-FR",
            "es-ES",
            "it-IT",
            "ja-JP",
            "zh-CN",
            "nl-NL",
            "pt-BR",
            "ru-RU",
        ]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return supported audio container formats."""
        if self._entry.options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE) == STT_SERVICE_MAI:
            return [AudioFormats.WAV]
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return supported audio codecs."""
        if self._entry.options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE) == STT_SERVICE_MAI:
            return [AudioCodecs.PCM]
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return supported audio bit rates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return supported audio sample rates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return supported audio channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> SpeechResult:
        """Process incoming audio stream and transcribe to text.

        :param metadata: SpeechMetadata specifying format, codec, sample rate, language.
        :param stream: AsyncIterable streaming audio byte chunks.
        :return: SpeechResult object.
        """
        audio_chunks: list[bytes] = [chunk async for chunk in stream]
        raw_audio = b"".join(audio_chunks)
        if not raw_audio:
            return SpeechResult(text="", result=SpeechResultState.ERROR)

        options = dict(self._entry.options)
        is_ogg = metadata.format == AudioFormats.OGG and metadata.codec == AudioCodecs.OPUS
        if is_ogg and options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE) == STT_SERVICE_MAI:
            LOGGER.warning("MAI-Transcribe requires WAV audio; Ogg Opus is not supported by this service")
            return SpeechResult(text="", result=SpeechResultState.ERROR)

        if raw_audio.startswith((b"RIFF", b"OggS")) or is_ogg:
            wav_payload = raw_audio
        else:
            wav_payload = wrap_pcm_in_wav(
                pcm_bytes=raw_audio,
                sample_rate=metadata.sample_rate,
                channels=metadata.channel,
                bit_depth=metadata.bit_rate,
            )

        language = metadata.language or self._entry.options.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
        if options.get(CONF_STT_LANGUAGE_MODE, DEFAULT_STT_LANGUAGE_MODE) == "fixed":
            language = options.get(CONF_STT_LANGUAGE, language)
        profanity_mode = self._entry.options.get(CONF_PROFANITY_MODE, DEFAULT_PROFANITY_MODE)

        res: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                res = await self.coordinator.async_transcribe_audio(
                    audio=wav_payload,
                    language=language,
                    profanity_mode=profanity_mode,
                    options=options,
                )
                break
            except (AzureSpeechApiClientCommunicationError, TimeoutError) as err:
                if attempt == 0:
                    LOGGER.debug("Transient STT error, retrying once: %s", err)
                    await asyncio.sleep(0.5)
                    continue

                LOGGER.warning("STT failed after retry: %s", err)
                return SpeechResult(text="", result=SpeechResultState.ERROR)
            except AzureSpeechApiClientAuthenticationError:
                self._entry.async_start_reauth(self.hass)
                return SpeechResult(text="", result=SpeechResultState.ERROR)
            except AzureSpeechApiClientError as err:
                LOGGER.warning("STT failed: %s", err)
                return SpeechResult(text="", result=SpeechResultState.ERROR)

        if res is None:
            return SpeechResult(text="", result=SpeechResultState.ERROR)

        status = res.get("RecognitionStatus")
        display_text = res.get("DisplayText", "")

        if status == "Success":
            return SpeechResult(text=display_text, result=SpeechResultState.SUCCESS)
        if status in ("NoMatch", "InitialSilenceTimeout"):
            return SpeechResult(text="", result=SpeechResultState.SUCCESS)

        return SpeechResult(text="", result=SpeechResultState.ERROR)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: AzureSpeechConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Azure Speech STT platform."""
    async_add_entities([AzureSpeechSTTEntity(entry)])
