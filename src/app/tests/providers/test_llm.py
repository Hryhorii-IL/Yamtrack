import json

from django.test import TestCase, override_settings

from app.providers import llm

DUNE = {"title": "Dune", "year": 2021}


def response_text(*items):
    """Return the JSON text of a RecommendationList."""
    return json.dumps({"items": list(items)})


class ItemStreamParserTests(TestCase):
    """Test extracting recommendations from partial JSON."""

    def feed_all(self, *chunks):
        """Feed chunks and return the yielded items per chunk."""
        parser = llm.ItemStreamParser()
        return [list(parser.feed(chunk)) for chunk in chunks]

    def test_items_yielded_when_complete(self):
        """Yield each item as soon as its closing brace arrives."""
        text = response_text(DUNE, DUNE)
        split = text.index("}") + 1

        self.assertEqual(
            self.feed_all(text[: split - 1], text[split - 1 : split], text[split:]),
            [[], [DUNE], [DUNE]],
        )

    def test_one_character_chunks(self):
        """Handle chunks split anywhere, including inside strings."""
        text = response_text(DUNE, {"title": "A, [b]}", "year": 1})

        items = [item for chunk in self.feed_all(*text) for item in chunk]

        self.assertEqual(
            items,
            [DUNE, {"title": "A, [b]}", "year": 1}],
        )

    def test_invalid_item_skipped(self):
        """Skip items that don't match the schema."""
        text = response_text({"year": 2020}, DUNE)

        self.assertEqual(self.feed_all(text), [[DUNE]])


class ProviderSelectionTests(TestCase):
    """Test choosing the AI provider from settings."""

    @override_settings(AI_PROVIDER="claude", ANTHROPIC_API_KEY="key")
    def test_claude_configured(self):
        """Use the Anthropic API key when Claude is selected."""
        self.assertTrue(llm.is_configured())
        self.assertEqual(llm.provider_label(), "Claude")
        self.assertEqual(llm.api_key_setting(), "ANTHROPIC_API_KEY")

    @override_settings(AI_PROVIDER="claude", ANTHROPIC_API_KEY="")
    def test_claude_without_key(self):
        """Ignore the Gemini key when Claude is selected."""
        self.assertFalse(llm.is_configured())

    @override_settings(AI_PROVIDER="gemini", ANTHROPIC_API_KEY="")
    def test_gemini_configured(self):
        """Use the Gemini API key when Gemini is selected."""
        self.assertTrue(llm.is_configured())
        self.assertEqual(llm.provider_label(), "Gemini")
