from unittest import TestCase

from app.ai_recommendations.bl.calculations.thinking import uses_thinking
from app.ai_recommendations.bl.types.thinking import ThinkingMode


class UsesThinkingTests(TestCase):
    """Test when a generation thinks."""

    def test_modes(self):
        """Think never, only in the background, or always."""
        cases = {
            (ThinkingMode.OFF, False): False,
            (ThinkingMode.OFF, True): False,
            (ThinkingMode.BACKGROUND, False): False,
            (ThinkingMode.BACKGROUND, True): True,
            (ThinkingMode.ALWAYS, False): True,
            (ThinkingMode.ALWAYS, True): True,
        }
        for (mode, background), expected in cases.items():
            with self.subTest(mode=mode, background=background):
                self.assertEqual(uses_thinking(mode, background=background), expected)
