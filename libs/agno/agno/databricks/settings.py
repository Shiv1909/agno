from typing import Any, Dict, Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agno.databricks.utils import normalize_host


class DatabricksSettings(BaseSettings):
    host: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_HOST"),
    )
    workspace_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_WORKSPACE_URL"),
    )
    token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_TOKEN", "DATABRICKS_PAT"),
    )
    client_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_CLIENT_ID"),
    )
    client_secret: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_CLIENT_SECRET"),
    )
    account_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_ACCOUNT_ID"),
    )
    timeout: float = Field(
        default=60.0,
        validation_alias=AliasChoices("DATABRICKS_TIMEOUT"),
    )
    max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices("DATABRICKS_MAX_RETRIES"),
    )
    default_headers: Dict[str, str] = Field(default_factory=dict)
    user_agent: str = "agno-databricks/0.1"

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("host", "workspace_url", mode="before")
    def validate_host(cls, value: Optional[str]) -> Optional[str]:
        return normalize_host(value)

    @field_validator("timeout")
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout must be greater than 0")
        return value

    @field_validator("max_retries")
    def validate_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        return value

    @model_validator(mode="after")
    def resolve_workspace_url(self) -> "DatabricksSettings":
        normalized_host = self.host or self.workspace_url
        self.host = normalized_host
        self.workspace_url = normalized_host
        return self

    @property
    def base_url(self) -> Optional[str]:
        return self.host

    @property
    def has_pat_auth(self) -> bool:
        return self.token is not None and self.token.strip() != ""

    @property
    def has_oauth_client_credentials(self) -> bool:
        return all([self.client_id, self.client_secret])

    @classmethod
    def from_values(cls, **values: Any) -> "DatabricksSettings":
        payload = {key: value for key, value in values.items() if value is not None}
        return cls.model_validate(payload)

    def with_overrides(self, **overrides: Any) -> "DatabricksSettings":
        payload = self.model_dump()
        normalized_overrides = {key: value for key, value in overrides.items() if value is not None}

        if "default_headers" in normalized_overrides:
            payload["default_headers"] = {
                **payload.get("default_headers", {}),
                **normalized_overrides.pop("default_headers"),
            }

        payload.update(normalized_overrides)
        return self.__class__.from_values(**payload)
