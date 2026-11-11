"""Config flow handler for azure_speech integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from custom_components.azure_speech.api import (
    AzureSpeechApiClientAuthenticationError,
    AzureSpeechApiClientCommunicationError,
    AzureSpeechApiClientError,
)
from custom_components.azure_speech.api.stt_options import validate_stt_options
from custom_components.azure_speech.const import (
    CONF_API_KEY,
    CONF_ENDPOINT,
    CONF_REGION,
    CONF_STT_SERVICE,
    DEFAULT_STT_SERVICE,
    DOMAIN,
    LOGGER,
    STT_SERVICE_FAST,
    STT_SERVICE_LLM,
    STT_SERVICE_MAI,
    STT_SERVICE_REALTIME,
    STT_SERVICE_SHORT_AUDIO,
)
from homeassistant.config_entries import ConfigEntry, ConfigFlow, FlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .schemas import get_options_schema, get_user_schema
from .schemas.stt import get_stt_options_schema, get_stt_service_schema
from .validators import validate_auth_input


class AzureSpeechConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Azure Speech."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial user setup step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            api_key = user_input[CONF_API_KEY]
            region = user_input[CONF_REGION]
            endpoint = user_input.get(CONF_ENDPOINT)

            unique_id = f"azure_speech_{region}"
            if endpoint:
                unique_id = f"azure_speech_{endpoint}"

            await self.async_set_unique_id(unique_id)
            self._abort_if_unique_id_configured()

            try:
                session = async_get_clientsession(self.hass)
                info = await validate_auth_input(
                    session=session,
                    api_key=api_key,
                    region=region,
                    endpoint=endpoint,
                )
                return self.async_create_entry(
                    title=info["title"],
                    data=user_input,
                )
            except AzureSpeechApiClientAuthenticationError:
                errors["base"] = "invalid_auth"
            except AzureSpeechApiClientCommunicationError:
                errors["base"] = "cannot_connect"
            except AzureSpeechApiClientError:
                errors["base"] = "unknown"
            except Exception:  # noqa: BLE001
                LOGGER.exception("Unexpected exception during Azure Speech config flow")
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=get_user_schema(),
            errors=errors,
        )

    async def async_step_reauth(self, entry_data: dict[str, Any]) -> FlowResult:
        """Perform reauth upon an authentication error."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Confirm reauth dialog."""
        errors: dict[str, str] = {}
        reauth_entry = self._get_reauth_entry()

        if user_input is not None:
            api_key = user_input[CONF_API_KEY]
            region = user_input.get(CONF_REGION, reauth_entry.data.get(CONF_REGION, "eastus"))
            endpoint = user_input.get(CONF_ENDPOINT, reauth_entry.data.get(CONF_ENDPOINT))

            try:
                session = async_get_clientsession(self.hass)
                await validate_auth_input(
                    session=session,
                    api_key=api_key,
                    region=region,
                    endpoint=endpoint,
                )
                return self.async_update_reload_and_abort(
                    reauth_entry,
                    data_updates={
                        CONF_API_KEY: api_key,
                        CONF_REGION: region,
                        CONF_ENDPOINT: endpoint,
                    },
                )
            except AzureSpeechApiClientAuthenticationError:
                errors["base"] = "invalid_auth"
            except AzureSpeechApiClientCommunicationError:
                errors["base"] = "cannot_connect"
            except Exception:  # noqa: BLE001
                LOGGER.exception("Unexpected exception during Azure Speech reauth flow")
                errors["base"] = "unknown"

        schema = get_user_schema(
            default_region=reauth_entry.data.get(CONF_REGION, "eastus"),
            default_endpoint=reauth_entry.data.get(CONF_ENDPOINT, ""),
        )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=schema,
            errors=errors,
            description_placeholders={"name": reauth_entry.title},
        )

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle reconfiguration of an existing entry."""
        errors: dict[str, str] = {}
        reconfig_entry = self._get_reconfigure_entry()

        if user_input is not None:
            api_key = user_input[CONF_API_KEY]
            region = user_input[CONF_REGION]
            endpoint = user_input.get(CONF_ENDPOINT)

            try:
                session = async_get_clientsession(self.hass)
                await validate_auth_input(
                    session=session,
                    api_key=api_key,
                    region=region,
                    endpoint=endpoint,
                )
                return self.async_update_reload_and_abort(
                    reconfig_entry,
                    data_updates=user_input,
                )
            except AzureSpeechApiClientAuthenticationError:
                errors["base"] = "invalid_auth"
            except AzureSpeechApiClientCommunicationError:
                errors["base"] = "cannot_connect"
            except Exception:  # noqa: BLE001
                LOGGER.exception("Unexpected exception during Azure Speech reconfigure flow")
                errors["base"] = "unknown"

        schema = self.add_suggested_values_to_schema(
            get_user_schema(),
            reconfig_entry.data,
        )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow handler for Azure Speech."""
        return AzureSpeechOptionsFlowHandler()


class AzureSpeechOptionsFlowHandler(OptionsFlow):
    """Handle options for Azure Speech integration."""

    def __init__(self) -> None:
        """Keep unsaved settings isolated until the selected page is submitted."""
        super().__init__()
        self._options: dict[str, Any] = {}

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Choose between speech synthesis and speech recognition settings."""
        self._options = dict(self.config_entry.options)
        return self.async_show_menu(step_id="init", menu_options=["tts", "stt"])

    async def async_step_tts(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage TTS defaults while preserving recognition settings."""
        schema = self._tts_schema()
        if user_input is not None:
            self._merge_options(schema, user_input)
            return self.async_create_entry(title="", data=self._options)

        return self.async_show_form(step_id="tts", data_schema=schema)

    def _tts_schema(self) -> vol.Schema:
        """Build the existing voice settings using cached Azure voices."""
        voices_list = []
        if hasattr(self.config_entry, "runtime_data") and self.config_entry.runtime_data:
            voices_list = getattr(self.config_entry.runtime_data.coordinator, "voices", [])

        return get_options_schema(
            voices_list=voices_list,
            current_options=self._options,
        )

    async def async_step_stt(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Choose the STT service before configuring its supported features."""
        errors: dict[str, str] = {}
        if user_input is not None:
            service = user_input.get(CONF_STT_SERVICE, DEFAULT_STT_SERVICE)
            if service in (
                STT_SERVICE_SHORT_AUDIO,
                STT_SERVICE_REALTIME,
                STT_SERVICE_FAST,
                STT_SERVICE_LLM,
                STT_SERVICE_MAI,
            ):
                self._options[CONF_STT_SERVICE] = service
                return await self.async_step_stt_options()
            errors[CONF_STT_SERVICE] = "invalid_stt_service"

        return self.async_show_form(
            step_id="stt",
            data_schema=get_stt_service_schema(self._options),
            errors=errors,
        )

    async def async_step_stt_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Validate and save service-specific recognition settings."""
        schema = get_stt_options_schema(self._options)
        errors: dict[str, str] = {}
        if user_input is not None:
            self._merge_options(schema, user_input)
            errors = validate_stt_options(self._options)
            if not errors:
                return self.async_create_entry(title="", data=self._options)

        return self.async_show_form(
            step_id="stt_options",
            data_schema=get_stt_options_schema(self._options),
            errors=errors,
        )

    def _merge_options(self, schema: vol.Schema, user_input: dict[str, Any]) -> None:
        """Replace the fields on this page, including cleared optional values."""
        for key in schema.schema:
            self._options.pop(key.schema, None)
        self._options.update(user_input)
