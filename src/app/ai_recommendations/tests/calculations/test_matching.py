from unittest import TestCase

from app.ai_recommendations.bl.calculations.matching import best_match, normalize_title
from app.ai_recommendations.bl.types.recommendation import RecommendedTitle
from app.ai_recommendations.tests.fakes import resolved


class BestMatchTests(TestCase):
    """Test matching recommended titles to search results."""

    def test_prefers_matching_year(self):
        """Pick the result of the recommended year among same titles."""
        results = [resolved("1", "Dune", 1984), resolved("2", "Dune", 2021)]

        match = best_match(results, RecommendedTitle(title="Dune", year=2021))

        self.assertEqual(match, results[1])

    def test_ignores_punctuation_and_accents(self):
        """Match titles that differ only in punctuation, case or accents."""
        results = [resolved("1", "Other"), resolved("2", "Amélie")]

        match = best_match(results, RecommendedTitle(title="amelie!"))

        self.assertEqual(match, results[1])

    def test_falls_back_to_same_title(self):
        """Use a same-title result when no year matches."""
        results = [resolved("1", "Dune", 1984)]

        match = best_match(results, RecommendedTitle(title="Dune", year=2021))

        self.assertEqual(match, results[0])

    def test_requires_exact_title(self):
        """Skip recommendations without a result of the same title."""
        results = [resolved("1", "First"), resolved("2", "Second")]

        self.assertIsNone(best_match(results, RecommendedTitle(title="Unknown")))
        self.assertIsNone(best_match([], RecommendedTitle(title="Unknown")))

    def test_normalize_title(self):
        """Keep only lowercase ASCII letters and digits."""
        self.assertEqual(normalize_title("Pokémon: Red & Blue 2"), "pokemonredblue2")
