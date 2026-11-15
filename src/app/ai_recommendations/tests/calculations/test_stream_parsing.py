import json
from unittest import TestCase

from app.ai_recommendations.bl.calculations.stream_parsing import ItemStreamParser
from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

DUNE = {"title": "Dune", "year": 2021}


def response_text(*items):
    """Return the JSON text of a recommendation list."""
    return json.dumps({"items": list(items)})


def feed_all(*chunks):
    """Feed chunks and return the yielded titles per chunk."""
    parser = ItemStreamParser()
    return [list(parser.feed(chunk)) for chunk in chunks]


class ItemStreamParserTests(TestCase):
    """Test extracting recommendations from partial JSON."""

    def test_items_yielded_when_complete(self):
        """Yield each item as soon as its closing brace arrives."""
        text = response_text(DUNE, DUNE)
        split = text.index("}") + 1
        dune = RecommendedTitle(title="Dune", year=2021)

        self.assertEqual(
            feed_all(text[: split - 1], text[split - 1 : split], text[split:]),
            [[], [dune], [dune]],
        )

    def test_one_character_chunks(self):
        """Handle chunks split anywhere, including inside strings."""
        text = response_text(DUNE, {"title": "A, [b]}", "year": 1})

        titles = [title for chunk in feed_all(*text) for title in chunk]

        self.assertEqual(
            titles,
            [
                RecommendedTitle(title="Dune", year=2021),
                RecommendedTitle(title="A, [b]}", year=1),
            ],
        )

    def test_item_without_title_skipped(self):
        """Skip items that have no title."""
        text = response_text({"year": 2020}, DUNE)

        self.assertEqual(feed_all(text), [[RecommendedTitle(title="Dune", year=2021)]])

    def test_invalid_year_dropped(self):
        """Keep the title of an item with an unusable year."""
        text = response_text({"title": "Dune", "year": "soon"})

        self.assertEqual(feed_all(text), [[RecommendedTitle(title="Dune")]])
