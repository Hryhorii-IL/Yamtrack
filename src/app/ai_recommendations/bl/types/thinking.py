from __future__ import annotations

from enum import StrEnum


class ThinkingMode(StrEnum):
    """When the LLM may think before answering, trading speed for quality."""

    OFF = "off"
    # Only where nobody waits for the answer: pool extensions, nightly refreshes
    BACKGROUND = "background"
    ALWAYS = "always"
