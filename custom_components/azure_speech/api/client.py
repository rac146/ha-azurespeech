"""Azure Speech API client implementation using aiohttp."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Final

import aiohttp

from custom_components.azure_speech.const import (
    CONF_STT_ENDPOINT,
    CONF_STT_LANGUAGE_MODE,
    CONF_STT_SERVICE,
    CONF_STT_TEXT_FORMAT,
    CONF_STT_TIMEOUT,
    DEFAULT_STT_LANGUAGE_MODE,
    DEFAULT_STT_SERVICE,
    DEFAULT_STT_TIMEOUT,
    LOGGER,
    STT_API_VERSION,
    STT_SERVICE_REALTIME,
    STT_SERVICE_SHORT_AUDIO,
)
from homeassistant.helpers.redact import async_redact_data

from .exceptions import (
    AzureSpeechApiClientAuthenticationError,
    AzureSpeechApiClientCommunicationError,
    AzureSpeechApiClientError,
    AzureSpeechApiClientRateLimitError,
)
from .realtime_stt import build_realtime_context, realtime_url, transcribe_realtime_audio
from .stt_options import build_transcription_definition, validate_stt_options

TIMEOUT: Final = 15


class AzureSpeechApiClient:
    """API client for communicating with Azure Speech REST and realtime services."""

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
        self._last_stt_request: dict[str, Any] | None = None
        self._last_stt_response: dict[str, Any] | None = None

    @property
    def last_stt_request(self) -> dict[str, Any] | None:
        """Return the last request settings, with phrase hints and instructions redacted."""
        return self._last_stt_request

    @property
    def last_stt_response(self) -> dict[str, Any] | None:
        """Return response metadata without recording the spoken text."""
        return self._last_stt_response

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
        """Transcribe WAV/Opus audio with the selected Azure recognition service.

        :param wav_audio_bytes: Binary WAV PCM 16kHz or Ogg Opus audio stream.
        :param language: Language BCP-47 locale tag (e.g. en-US).
        :param profanity_mode: Profanity filter setting (masked, removed, raw).
        :param stt_options: Optional service-specific recognition settings.
        :return: Dict containing RecognitionStatus and DisplayText.
        """
        options = stt_options or {}
        self._last_stt_request = None
        self._last_stt_response = None
        if errors := validate_stt_options(options):
            raise AzureSpeechApiClientError(f"Invalid STT settings: {', '.join(errors)}")
        if profanity_mode not in ("masked", "removed", "raw"):
            raise AzureSpeechApiClientError("Invalid STT profanity filter")

        service = options.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE)
        endpoint = options.get(CONF_STT_ENDPOINT) or self._endpoint
        is_ogg = wav_audio_bytes.startswith(b"OggS")
        content_type = "audio/ogg; codecs=opus" if is_ogg else "audio/wav; codecs=audio/pcm; samplerate=16000"
        headers = self._headers({"Accept": "application/json"})
        if service == STT_SERVICE_REALTIME:
            context = build_realtime_context(options, language, profanity_mode)
            realtime_params = {
                "language": language,
                "profanity": profanity_mode,
                "format": "detailed" if options.get(CONF_STT_TEXT_FORMAT, "display") == "lexical" else "simple",
            }
            self._last_stt_request = {
                "service": service,
                "transport": "websocket",
                "region": self._region,
                "custom_endpoint": bool(endpoint),
                "language_mode": options.get(CONF_STT_LANGUAGE_MODE, DEFAULT_STT_LANGUAGE_MODE),
                "query": dict(realtime_params),
                "context": async_redact_data(context, {"items"}),
                "audio_format": "wav_pcm",
                "audio_bytes": len(wav_audio_bytes),
            }
            LOGGER.debug("Azure STT request settings: %s", self._last_stt_request)
            realtime_result, realtime_metadata = await transcribe_realtime_audio(
                session=self._session,
                url=realtime_url(endpoint or self.stt_base_url),
                headers=headers,
                params=realtime_params,
                audio=wav_audio_bytes,
                context=context,
                timeout=options.get(CONF_STT_TIMEOUT, DEFAULT_STT_TIMEOUT),
            )
            self._last_stt_response = realtime_metadata
            LOGGER.debug("Azure STT response metadata: %s", realtime_metadata)
            return realtime_result

        params: dict[str, str]
        payload: bytes | aiohttp.FormData
        definition: dict[str, Any] | None = None

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

        self._last_stt_request = {
            "service": service,
            "region": self._region,
            "custom_endpoint": bool(endpoint),
            "language_mode": options.get(CONF_STT_LANGUAGE_MODE, DEFAULT_STT_LANGUAGE_MODE),
            "query": dict(params),
            "definition": async_redact_data(definition, {"phrases", "prompt"}),
            "audio_format": "ogg_opus" if is_ogg else "wav_pcm",
            "audio_bytes": len(wav_audio_bytes),
        }
        LOGGER.debug("Azure STT request settings: %s", self._last_stt_request)

        response_metadata: dict[str, Any] | None = None
        try:
            async with asyncio.timeout(options.get(CONF_STT_TIMEOUT, DEFAULT_STT_TIMEOUT)):
                response = await self._session.post(url, headers=headers, params=params, data=payload)
                try:
                    response_metadata = {
                        "http_status": response.status,
                        "request_id": response.headers.get("apim-request-id") or response.headers.get("x-ms-request-id"),
                        "redirect_count": len(response.history),
                    }
                    self._last_stt_response = response_metadata
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

        if response_metadata is not None:
            response_metadata["response_fields"] = sorted(result)
            phrases = result.get("phrases", [])
            if isinstance(phrases, list):
                phrase_results = [phrase for phrase in phrases if isinstance(phrase, dict)]
                response_metadata["phrase_count"] = len(phrase_results)
                response_metadata["phrase_fields"] = sorted({key for phrase in phrase_results for key in phrase})
                confidences = [
                    phrase["confidence"]
                    for phrase in phrase_results
                    if isinstance(phrase.get("confidence"), (int, float))
                ]
                response_metadata["confidence_count"] = len(confidences)
                response_metadata["confidence_min"] = min(confidences) if confidences else None
                response_metadata["confidence_max"] = max(confidences) if confidences else None
            self._last_stt_response = response_metadata
            LOGGER.debug("Azure STT response metadata: %s", response_metadata)

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
