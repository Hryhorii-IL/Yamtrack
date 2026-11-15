from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ProviderInfo:
    """The selected AI provider, as the preferences page describes it."""

    label: str
    api_key_setting: str
    configured: bool
