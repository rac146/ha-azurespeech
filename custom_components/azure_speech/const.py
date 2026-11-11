"""Constants for azure_speech."""

from logging import Logger, getLogger
from typing import Final

from homeassistant.const import Platform

LOGGER: Logger = getLogger(__package__)

# Integration metadata
DOMAIN: Final = "azure_speech"
ATTRIBUTION: Final = "Data provided by Microsoft Azure Speech Services"

# Configuration options
CONF_API_KEY: Final = "api_key"
CONF_REGION: Final = "region"
CONF_ENDPOINT: Final = "endpoint"
CONF_VOICE: Final = "voice"
CONF_STYLE: Final = "style"
CONF_STYLE_DEGREE: Final = "style_degree"
CONF_LANGUAGE: Final = "language"
CONF_PITCH: Final = "pitch"
CONF_RATE: Final = "rate"
CONF_VOLUME: Final = "volume"
CONF_AUDIO_FORMAT: Final = "audio_format"
CONF_PROFANITY_MODE: Final = "profanity_mode"
CONF_STT_SERVICE: Final = "stt_service"
CONF_STT_ENDPOINT: Final = "stt_endpoint"
CONF_STT_LANGUAGE_MODE: Final = "stt_language_mode"
CONF_STT_LANGUAGE: Final = "stt_language"
CONF_STT_CANDIDATE_LANGUAGES: Final = "stt_candidate_languages"
CONF_STT_PHRASE_LIST: Final = "stt_phrase_list"
CONF_STT_PHRASE_LIST_WEIGHT: Final = "stt_phrase_list_weight"
CONF_STT_POST_REFINEMENT: Final = "stt_post_refinement"
CONF_STT_PROMPT: Final = "stt_prompt"
CONF_STT_TASK: Final = "stt_task"
CONF_STT_TARGET_LANGUAGE: Final = "stt_target_language"
CONF_STT_TRANSCRIPT_STYLE: Final = "stt_transcript_style"
CONF_STT_TEXT_FORMAT: Final = "stt_text_format"
CONF_STT_TIMEOUT: Final = "stt_timeout"

# Default configuration values
DEFAULT_REGION: Final = "eastus"
DEFAULT_LANGUAGE: Final = "en-US"
DEFAULT_VOICE: Final = "en-US-AvaMultilingualNeural"
DEFAULT_STYLE: Final = "default"
DEFAULT_STYLE_DEGREE: Final = ""
DEFAULT_PITCH: Final = "default"
DEFAULT_RATE: Final = "default"
DEFAULT_VOLUME: Final = "default"
DEFAULT_AUDIO_FORMAT: Final = "mp3"
DEFAULT_PROFANITY_MODE: Final = "masked"
DEFAULT_STT_SERVICE: Final = "short_audio"
DEFAULT_STT_LANGUAGE_MODE: Final = "pipeline"
DEFAULT_STT_POST_REFINEMENT: Final = False
DEFAULT_STT_TIMEOUT: Final = 15

# STT service names and limits.
STT_SERVICE_SHORT_AUDIO: Final = "short_audio"
STT_SERVICE_REALTIME: Final = "realtime"
STT_SERVICE_FAST: Final = "fast_transcription"
STT_SERVICE_LLM: Final = "llm_speech"
STT_SERVICE_MAI: Final = "mai_transcribe"
STT_API_VERSION: Final = "2025-10-15"
STT_MAX_PHRASES: Final = 2000
STT_MAX_PROMPT_LENGTH: Final = 20000

# Common Azure Speech regions
AZURE_REGIONS: Final[list[str]] = [
    "eastus",
    "eastus2",
    "westus",
    "westus2",
    "westus3",
    "centralus",
    "northcentralus",
    "southcentralus",
    "westeurope",
    "northeurope",
    "uksouth",
    "ukwest",
    "francecentral",
    "germanywestcentral",
    "eastasia",
    "southeastasia",
    "japaneast",
    "japanwest",
    "australiaeast",
    "centralindia",
    "brazilsouth",
    "canadacentral",
    "koreacentral",
]

# Audio Output Format mappings for Azure TTS REST API
AZURE_OUTPUT_FORMATS: Final[dict[str, str]] = {
    "mp3": "audio-24khz-48kbitrate-mono-mp3",
    "wav": "riff-24khz-16bit-mono-pcm",
    "ogg": "webm-24khz-16bit-24kbps-mono-opus",
}

# Platforms supported
PLATFORMS: Final[list[Platform]] = [
    Platform.TTS,
    Platform.STT,
    Platform.SENSOR,
]
