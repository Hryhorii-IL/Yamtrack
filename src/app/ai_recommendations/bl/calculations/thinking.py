from __future__ import annotations

from app.ai_recommendations.bl.types.thinking import ThinkingMode


def uses_thinking(mode: ThinkingMode, *, background: bool) -> bool:
    """Return whether a generation thinks, given whether anyone waits for it."""
    if mode is ThinkingMode.ALWAYS:
        return True
    return mode is ThinkingMode.BACKGROUND and background
