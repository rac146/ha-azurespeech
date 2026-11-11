"""Azure Speech API client implementation using aiohttp."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Final

import aiohttp

from custom_components.azure_speech.const import (
    CONF_STT_ENDPOINT,
    CONF_STT_SERVICE,
    CONF_STT_TEXT_FORMAT,
    CONF_STT_TIMEOUT,
    DEFAULT_STT_SERVICE,
    DEFAULT_STT_TIMEOUT,
    STT_API_VERSION,
    STT_SERVICE_SHORT_AUDIO,
)

from .exceptions import (
    AzureSpeechApiClientAuthenticationError,
    AzureSpeechApiClientCommunicationError,
    AzureSpeechApiClientError,
    AzureSpeechApiClientRateLimitError,
)
from .stt_options import build_transcription_definition, validate_stt_options

TIMEOUT: Final = 15


class AzureSpeechApiClient:
    """API client for communicating with Azure Speech REST APIs."""

    def __init__(
        self,
        api_key: str,
        region: str,
        session: aiohttp.ClientSession,
        endpoint: str | None = None,
    ) -> None:
        """Initialize the Azure Speech API client.

        :param api_key: Azure Cognitive Services API Key.
        :param region: Azure Region (e.g. eastus, westeurope).
        :param session: aiohttp ClientSession provided by Home Assistant.
        :param endpoint: Optional custom endpoint base URL.
        """
        self._api_key = api_key
        self._region = region
        self._session = session
        self._endpoint = endpoint.rstrip("/") if endpoint else None

    @property
    def tts_base_url(self) -> str:
        """Get the base URL for Azure TTS services."""
        if self._endpoint:
            return self._endpoint
        return f"https://{self._region}.tts.speech.microsoft.com"

    @property
    def stt_base_url(self) -> str:
        """Get the base URL for Azure STT services."""
        if self._endpoint:
            return self._endpoint
        return f"https://{self._region}.stt.speech.microsoft.com"

    def _headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        """Generate base headers for API requests."""
        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key,
            "User-Agent": "HomeAssistant-AzureSpeechIntegration/0.1.0",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    async def _handle_response_errors(self, response: aiohttp.ClientResponse) -> None:
        """Handle HTTP error status codes and raise integration exceptions."""
        if response.status in (401, 403):
            raise AzureSpeechApiClientAuthenticationError(
                f"Authentication failed (HTTP {response.status}): Invalid API key or region."
            )
        if response.status == 429:
            retry_after_hdr = response.headers.get("Retry-After")
            retry_after = int(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else 60
            raise AzureSpeechApiClientRateLimitError(retry_after=retry_after)
        if response.status >= 400:
            text = await response.text()
            raise AzureSpeechApiClientError(f"Azure Speech API error (HTTP {response.status}): {text}")

    async def get_voices(self) -> list[dict[str, Any]]:
        """Fetch the list of available neural voices for the configured region."""
        url = f"{self.tts_base_url}/cognitiveservices/voices/list"
        try:
            async with asyncio.timeout(TIMEOUT):
                response = await self._session.get(url, headers=self._headers())
                await self._handle_response_errors(response)
                voices: list[dict[str, Any]] = await response.json()
                return voices
        except TimeoutError as err:
            raise AzureSpeechApiClientCommunicationError(
                "Timeout while fetching voices list from Azure Speech API"
            ) from err
        except aiohttp.ClientError as err:
            raise AzureSpeechApiClientCommunicationError(f"Connection error fetching voices: {err}") from err

    async def generate_tts_audio(
        self,
        ssml_content: str,
        output_format: str = "audio-24khz-48kbitrate-mono-mp3",
    ) -> bytes:
        """Request TTS synthesis audio bytes from Azure Speech REST API.

        :param ssml_content: SSML XML payload string.
        :param output_format: X-Microsoft-OutputFormat header value.
        :return: Raw binary audio bytes (MP3, WAV, OPUS).
        """
        url = f"{self.tts_base_url}/cognitiveservices/v1"
        headers = self._headers(
            {
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": output_format,
            }
        )
        try:
            async with asyncio.timeout(TIMEOUT):
                response = await self._session.post(
                    url,
                    headers=headers,
                    data=ssml_content.encode("utf-8"),
                )
                await self._handle_response_errors(response)
                return await response.read()
        except TimeoutError as err:
            raise AzureSpeechApiClientCommunicationError("Timeout generating TTS audio from Azure Speech API") from err
        except aiohttp.ClientError as err:
            raise AzureSpeechApiClientCommunicationError(f"Connection error generating TTS audio: {err}") from err

    async def transcribe_stt_audio(
        self,
        wav_audio_bytes: bytes,
        language: str = "en-US",
        profanity_mode: str = "masked",
        *,
        stt_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Transcribe WAV/Opus audio with the selected Azure REST service.

        :param wav_audio_bytes: Binary WAV PCM 16kHz or Ogg Opus audio stream.
        :param language: Language BCP-47 locale tag (e.g. en-US).
        :param profanity_mode: Profanity filter setting (masked, removed, raw).
        :param stt_options: Optional service-specific recognition settings.
        :return: Dict containing RecognitionStatus and DisplayText.
        """
        options = stt_options or {}
        if errors := validate_stt_options(options):
            raise AzureSpeechApiClientError(f"Invalid STT settings: {', '.join(errors)}")
        if profanity_mode not in ("masked", "removed", "raw"):
            raise AzureSpeechApiClientError("Invalid STT profanity filter")

        service = options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE)
        endpoint = options.get(CONF_STT_ENDPOINT) or self._endpoint
        is_ogg = wav_audio_bytes.startswith(b"OggS")
        content_type = "audio/ogg; codecs=opus" if is_ogg else "audio/wav; codecs=audio/pcm; samplerate=16000"
        headers = self._headers({"Accept": "application/json"})
        params: dict[str, str]
        payload: bytes | aiohttp.FormData

        if service == STT_SERVICE_SHORT_AUDIO:
            base_url = endpoint.rstrip("/") if endpoint else self.stt_base_url
            if endpoint and ".cognitiveservices.azure.com" in base_url:
                base_url = f"{base_url}/stt"
            url = f"{base_url}/speech/recognition/conversation/cognitiveservices/v1"
            params = {"language": language, "profanity": profanity_mode}
            if options.get(CONF_STT_TEXT_FORMAT, "display") == "lexical":
                params["format"] = "detailed"
            headers["Content-Type"] = content_type
            payload = wav_audio_bytes
        else:
            base_url = endpoint.rstrip("/") if endpoint else f"https://{self._region}.api.cognitive.microsoft.com"
            url = f"{base_url}/speechtotext/transcriptions:transcribe"
            params = {"api-version": STT_API_VERSION}
            definition = build_transcription_definition(options, language, profanity_mode)
            payload = aiohttp.FormData()
            payload.add_field(
                "audio", wav_audio_bytes, filename="speech.ogg" if is_ogg else "speech.wav", content_type=content_type
            )
            payload.add_field("definition", json.dumps(definition), content_type="application/json")

        try:
            async with asyncio.timeout(options.get(CONF_STT_TIMEOUT, DEFAULT_STT_TIMEOUT)):
                response = await self._session.post(url, headers=headers, params=params, data=payload)
                try:
                    await self._handle_response_errors(response)
                    result = await response.json()
                finally:
                    response.release()
        except TimeoutError as err:
            raise AzureSpeechApiClientCommunicationError("Timeout transcribing audio with Azure STT API") from err
        except aiohttp.ClientError as err:
            raise AzureSpeechApiClientCommunicationError(f"Connection error transcribing STT audio: {err}") from err
        except ValueError as err:
            raise AzureSpeechApiClientError("Azure STT returned an invalid JSON response") from err

        if not isinstance(result, dict) or "error" in result:
            raise AzureSpeechApiClientError("Azure STT returned an invalid transcription response")
        if service == STT_SERVICE_SHORT_AUDIO:
            if result.get("RecognitionStatus") == "Success" and options.get(CONF_STT_TEXT_FORMAT) == "lexical":
                alternatives = result.get("NBest", [])
                if (
                    not isinstance(alternatives, list)
                    or not alternatives
                    or not isinstance(alternatives[0], dict)
                    or not isinstance(alternatives[0].get("Lexical"), str)
                ):
                    raise AzureSpeechApiClientError("Azure STT returned no lexical transcription")
                result["DisplayText"] = alternatives[0]["Lexical"]
            if result.get("RecognitionStatus") == "Success" and not isinstance(result.get("DisplayText"), str):
                raise AzureSpeechApiClientError("Azure STT returned no display transcription")
            return result

        combined_phrases = result.get("combinedPhrases")
        if not isinstance(combined_phrases, list) or any(
            not isinstance(phrase, dict) or not isinstance(phrase.get("text"), str) for phrase in combined_phrases
        ):
            raise AzureSpeechApiClientError("Azure STT returned no combined transcription")
        return {
            "RecognitionStatus": "Success",
            "DisplayText": " ".join(phrase["text"].strip() for phrase in combined_phrases if phrase["text"].strip()),
        }
