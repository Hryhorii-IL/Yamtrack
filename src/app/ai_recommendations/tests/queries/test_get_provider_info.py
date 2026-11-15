from django.test import SimpleTestCase, override_settings

from app.ai_recommendations.bl.queries.provider import get_provider_info
from app.ai_recommendations.bl.types.provider import ProviderInfo


class ProviderQueryTests(SimpleTestCase):
    """Test describing the selected AI provider."""

    @override_settings(AI_PROVIDER="claude", ANTHROPIC_API_KEY="")
    def test_claude_without_key(self):
        """Use the selected provider's label, key setting and key."""
        self.assertEqual(
            get_provider_info(),
            ProviderInfo(
                label="Claude",
                api_key_setting="ANTHROPIC_API_KEY",
                configured=False,
            ),
        )

    @override_settings(AI_PROVIDER="gemini", GEMINI_API_KEY="key")
    def test_gemini_configured(self):
        """Report a configured Gemini provider."""
        info = get_provider_info()

        self.assertEqual(info.label, "Gemini")
        self.assertTrue(info.configured)
