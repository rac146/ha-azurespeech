"""Standard Azure realtime STT using the Speech SDK's WebSocket protocol.

Microsoft's Python SDK has native dependencies unavailable on some HA runtimes.
This client uses the existing aiohttp session and the message framing and dynamic
grammar from Microsoft's JavaScript SDK (MIT licensed):
https://github.com/microsoft/cognitive-services-speech-sdk-js/tree/master/src/common.speech

PhraseListGrammar is encoded as speech.context.dgi, before any audio is sent.
Optional monolingual post-stream refinement uses the SDK's PostRefinement
setting and universal v2 endpoint. Only final phrase results are returned.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from io import BytesIO
import json
import struct
from typing import Any
from uuid import uuid4
import wave

import aiohttp
from yarl import URL

from custom_components.azure_speech.const import (
    CONF_STT_PHRASE_LIST,
    CONF_STT_PHRASE_LIST_WEIGHT,
    CONF_STT_POST_REFINEMENT,
    CONF_STT_TEXT_FORMAT,
    DEFAULT_STT_POST_REFINEMENT,
)

from .exceptions import (
    AzureSpeechApiClientAuthenticationError,
    AzureSpeechApiClientCommunicationError,
    AzureSpeechApiClientError,
    AzureSpeechApiClientRateLimitError,
)
from .stt_options import split_stt_list

PCM_BYTES_PER_SECOND = 32000
AUDIO_CHUNK_BYTES = 4096
MAX_AUDIO_SECONDS = 60


def build_realtime_context(options: dict[str, Any], language: str, profanity_mode: str) -> dict[str, Any]:
    """Encode phrase hints exactly as the SDK's DynamicGrammarBuilder does."""
    context: dict[str, Any] = {
        "phraseDetection": {
            "mode": "Conversation",
            "language": language,
            "enrichment": {"profanity": profanity_mode.capitalize()},
        }
    }
    if options.get(CONF_STT_POST_REFINEMENT, DEFAULT_STT_POST_REFINEMENT):
        context["phraseDetection"]["enrichment"]["conversation"] = {"postprocessingoption": "PostRefinement"}
    if options.get(CONF_STT_TEXT_FORMAT, "display") == "lexical":
        context["phraseOutput"] = {"format": "Detailed", "detailed": {"options": []}}
    phrases = split_stt_list(options.get(CONF_STT_PHRASE_LIST, ""))
    if phrases:
        context["dgi"] = {
            "groups": [{"type": "Generic", "items": [{"text": phrase} for phrase in phrases]}],
            "bias": options.get(CONF_STT_PHRASE_LIST_WEIGHT, 1.0),
        }
    return context


def realtime_url(base_url: str, *, post_refinement: bool = DEFAULT_STT_POST_REFINEMENT) -> str:
    """Use the SDK's universal v2 route for refinement, or the standard v1 route."""
    url = URL(base_url).with_scheme("wss")
    if post_refinement:
        return str(url.with_path("/stt/speech/universal/v2"))
    prefix = "/stt" if (url.host or "").endswith(".cognitiveservices.azure.com") else ""
    return str(url.with_path(f"{prefix}/speech/recognition/conversation/cognitiveservices/v1"))


def _message_headers(path: str, request_id: str, content_type: str | None = None) -> str:
    """Create Speech protocol headers with a UTC timestamp."""
    timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    headers = f"Path: {path}\r\nX-RequestId: {request_id}\r\nX-Timestamp: {timestamp}\r\n"
    if content_type:
        headers += f"Content-Type: {content_type}\r\n"
    return headers


def _audio_message(request_id: str, audio: bytes, content_type: str | None = None) -> bytes:
    """Prefix binary audio with the SDK's two-byte big-endian header length."""
    headers = _message_headers("audio", request_id, content_type).encode("ascii")
    return struct.pack(">H", len(headers)) + headers + audio


def _parse_message(message: str | bytes) -> tuple[dict[str, str], str]:
    """Separate response headers and body for text or framed binary messages."""
    try:
        if isinstance(message, bytes):
            if len(message) < 2:
                raise ValueError("Missing header length")
            header_length = int.from_bytes(message[:2], "big")
            if len(message) < header_length + 2:
                raise ValueError("Truncated headers")
            header_text = message[2 : header_length + 2].decode("ascii")
            body = message[header_length + 2 :].decode("utf-8")
        else:
            header_text, _, body = message.partition("\r\n\r\n")
        headers = {}
        for line in header_text.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                headers[key.strip().lower()] = value.strip()
        if "path" not in headers:
            raise ValueError("Missing message path")
        return headers, body
    except ValueError as err:
        raise AzureSpeechApiClientError("Azure realtime STT returned an invalid protocol message") from err


def _prepare_audio(audio: bytes) -> tuple[bytes, bytes]:
    """Read PCM from an in-memory WAV and create an SDK streaming WAV header."""
    try:
        with wave.open(BytesIO(audio), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise AzureSpeechApiClientError("Realtime STT requires mono, 16-bit, 16 kHz PCM WAV audio")
            frame_count = wav.getnframes()
            if frame_count > 16000 * MAX_AUDIO_SECONDS:
                raise AzureSpeechApiClientError("Realtime STT supports short commands up to 60 seconds")
            pcm = wav.readframes(frame_count)
            if len(pcm) != frame_count * 2:
                raise AzureSpeechApiClientError("Realtime STT received truncated WAV audio")
    except (wave.Error, EOFError) as err:
        raise AzureSpeechApiClientError("Realtime STT requires valid PCM WAV audio") from err
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 0, b"WAVE", b"fmt ", 16, 1, 1, 16000, PCM_BYTES_PER_SECOND, 2, 16, b"data", 0
    )
    return header, pcm


async def _send_audio(
    websocket: aiohttp.ClientWebSocketResponse,
    request_id: str,
    header: bytes,
    pcm: bytes,
    upload_complete: asyncio.Event,
) -> None:
    """Send WAV header, audio chunks, and an explicit empty end-of-stream frame."""
    await websocket.send_bytes(_audio_message(request_id, header, "audio/x-wav"))
    loop = asyncio.get_running_loop()
    started = loop.time()
    for offset in range(0, len(pcm), AUDIO_CHUNK_BYTES):
        # Match the SDK's five-second fast lane, then send at twice realtime speed.
        target_time = (offset - PCM_BYTES_PER_SECOND * 5) / (PCM_BYTES_PER_SECOND * 2)
        delay = target_time - (loop.time() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        await websocket.send_bytes(_audio_message(request_id, pcm[offset : offset + AUDIO_CHUNK_BYTES]))
    await websocket.send_bytes(_audio_message(request_id, b""))
    upload_complete.set()


def _phrase_text(phrase: dict[str, Any], text_format: str) -> str:
    """Use the final display or lexical result, preserving Azure's formatting."""
    if text_format == "lexical":
        alternatives = phrase.get("NBest")
        if not isinstance(alternatives, list) or not alternatives or not isinstance(alternatives[0], dict):
            raise AzureSpeechApiClientError("Azure realtime STT returned no lexical transcription")
        text = alternatives[0].get("Lexical")
    else:
        text = phrase.get("DisplayText")
    if not isinstance(text, str):
        raise AzureSpeechApiClientError("Azure realtime STT returned no transcription text")
    return text.strip()


async def _receive_result(
    websocket: aiohttp.ClientWebSocketResponse,
    request_id: str,
    text_format: str,
    upload_complete: asyncio.Event,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Collect final phrases and require a completed turn before returning text."""
    texts: list[str] = []
    statuses: list[str] = []
    async for message in websocket:
        if message.type == aiohttp.WSMsgType.ERROR:
            raise AzureSpeechApiClientCommunicationError("Azure realtime STT WebSocket failed")
        if message.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
            continue
        headers, body = _parse_message(message.data)
        if headers.get("x-requestid", request_id).lower() != request_id.lower():
            continue
        path = headers["path"].lower()
        if path == "speech.phrase":
            try:
                phrase = json.loads(body)
            except ValueError as err:
                raise AzureSpeechApiClientError("Azure realtime STT returned an invalid phrase response") from err
            if not isinstance(phrase, dict):
                raise AzureSpeechApiClientError("Azure realtime STT returned an invalid phrase response")
            status = phrase.get("RecognitionStatus")
            if status == "Success":
                texts.append(_phrase_text(phrase, text_format))
            elif status not in ("NoMatch", "InitialSilenceTimeout", "EndOfDictation"):
                raise AzureSpeechApiClientError("Azure realtime STT could not recognize the audio")
            statuses.append(status)
        elif path == "turn.end":
            if not upload_complete.is_set():
                raise AzureSpeechApiClientCommunicationError(
                    "Azure realtime STT ended before the full command was sent"
                )
            if not statuses:
                raise AzureSpeechApiClientError("Azure realtime STT ended without a recognition result")
            metadata.update(phrase_count=len(texts), recognition_statuses=statuses)
            return {
                "RecognitionStatus": "Success" if texts else "NoMatch",
                "DisplayText": " ".join(text for text in texts if text),
            }
    metadata["close_code"] = websocket.close_code
    if websocket.close_code in (1002, 1003, 1007, 1008, 4000):
        raise AzureSpeechApiClientError(f"Azure realtime STT rejected the request (WebSocket {websocket.close_code})")
    raise AzureSpeechApiClientCommunicationError("Azure realtime STT closed before recognition completed")


async def transcribe_realtime_audio(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    params: dict[str, str],
    audio: bytes,
    context: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recognize a buffered short command with phrase hints and bounded cleanup."""
    wav_header, pcm = _prepare_audio(audio)
    request_id = uuid4().hex
    metadata: dict[str, Any] = {"transport": "websocket", "request_id": request_id}
    if not pcm:
        return {"RecognitionStatus": "NoMatch", "DisplayText": ""}, metadata
    connection_headers = {**headers, "X-ConnectionId": uuid4().hex}
    try:
        async with asyncio.timeout(timeout):
            async with session.ws_connect(url, headers=connection_headers, params=params) as websocket:
                config = {
                    "context": {
                        "system": {"name": "HomeAssistant-AzureSpeech", "version": "0.1.0", "lang": "Python"}
                    }
                }
                for path, body in (("speech.config", config), ("speech.context", context)):
                    await websocket.send_str(
                        _message_headers(path, request_id, "application/json") + "\r\n" + json.dumps(body)
                    )
                upload_complete = asyncio.Event()
                sender = asyncio.create_task(_send_audio(websocket, request_id, wav_header, pcm, upload_complete))
                receiver = asyncio.create_task(
                    _receive_result(
                        websocket,
                        request_id,
                        "lexical" if params["format"] == "detailed" else "display",
                        upload_complete,
                        metadata,
                    )
                )
                try:
                    _, result = await asyncio.gather(sender, receiver)
                    return result, metadata
                finally:
                    sender.cancel()
                    receiver.cancel()
                    await asyncio.gather(sender, receiver, return_exceptions=True)
    except aiohttp.WSServerHandshakeError as err:
        if err.status in (401, 403):
            raise AzureSpeechApiClientAuthenticationError("Azure realtime STT authentication failed") from err
        if err.status == 429:
            retry_after = (err.headers or {}).get("Retry-After", "")
            raise AzureSpeechApiClientRateLimitError(int(retry_after) if retry_after.isdigit() else 60) from err
        if err.status >= 500:
            raise AzureSpeechApiClientCommunicationError(
                f"Azure realtime STT service unavailable (HTTP {err.status})"
            ) from err
        raise AzureSpeechApiClientError(f"Azure realtime STT handshake failed (HTTP {err.status})") from err
    except TimeoutError as err:
        raise AzureSpeechApiClientCommunicationError("Timeout transcribing audio with Azure realtime STT") from err
    except aiohttp.ClientError as err:
        raise AzureSpeechApiClientCommunicationError(
            "Connection error transcribing audio with Azure realtime STT"
        ) from err
