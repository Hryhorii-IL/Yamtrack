from unittest.mock import MagicMock, patch

import httpx
from django.conf import settings
from django.core.cache import cache
from django.test import TestCase, override_settings
from google.genai import errors

from app.providers import gemini, services


class GeminiTests(TestCase):
    """Test the Gemini client wrapper."""

    def setUp(self):
        """Clear models marked as unavailable."""
        cache.clear()

    @patch("app.providers.gemini.genai.Client")
    def test_recommend(self, mock_client):
        """Return the parsed recommendations as dicts."""
        response = MagicMock()
        response.parsed = gemini.RecommendationList(
            items=[gemini.Recommendation(title="Dune", year=2021, reason="Epic")],
        )
        mock_client.return_value.models.generate_content.return_value = response

        result = gemini.recommend("system", "prompt")

        self.assertEqual(result, [{"title": "Dune", "year": 2021, "reason": "Epic"}])
        kwargs = mock_client.return_value.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], settings.GEMINI_MODELS[0])
        self.assertEqual(kwargs["contents"], "prompt")
        self.assertEqual(kwargs["config"].system_instruction, "system")

    @patch("app.providers.gemini.genai.Client")
    def test_recommend_api_error(self, mock_client):
        """Wrap Gemini API errors in ProviderAPIError."""
        mock_client.return_value.models.generate_content.side_effect = errors.APIError(
            429, {"error": {"message": "quota exceeded"}}
        )

        with self.assertRaises(services.ProviderAPIError) as context:
            gemini.recommend("system", "prompt")

        self.assertEqual(context.exception.status_code, 429)
        self.assertIn("Gemini", str(context.exception))

    @patch("app.providers.gemini.genai.Client")
    def test_recommend_invalid_response(self, mock_client):
        """Raise ProviderAPIError when the response can't be parsed."""
        response = MagicMock()
        response.parsed = None
        mock_client.return_value.models.generate_content.return_value = response

        with self.assertRaises(services.ProviderAPIError):
            gemini.recommend("system", "prompt")

    @patch("app.providers.gemini.genai.Client")
    def test_recommend_network_error(self, mock_client):
        """Wrap network errors and timeouts in ProviderAPIError."""
        mock_client.return_value.models.generate_content.side_effect = (
            httpx.ConnectTimeout("timed out")
        )

        with self.assertRaises(services.ProviderAPIError) as context:
            gemini.recommend("system", "prompt")

        self.assertIsNone(context.exception.status_code)


@override_settings(GEMINI_MODELS=["first", "second", "third"])
class GeminiFallbackTests(TestCase):
    """Test falling back to the next configured Gemini model."""

    def setUp(self):
        """Create a successful response and clear unavailable models."""
        cache.clear()
        self.response = MagicMock()
        self.response.parsed = gemini.RecommendationList(
            items=[gemini.Recommendation(title="Dune", year=2021, reason="Epic")],
        )

    def called_models(self, mock_client):
        """Return the models passed to generate_content in order."""
        return [
            call.kwargs["model"]
            for call in mock_client.return_value.models.generate_content.call_args_list
        ]

    @patch("app.providers.gemini.genai.Client")
    def test_fallback_to_next_model(self, mock_client):
        """Try the next model when one fails."""
        mock_client.return_value.models.generate_content.side_effect = [
            errors.APIError(400, {"error": {"message": "bad model name"}}),
            httpx.ReadTimeout("timed out"),
            self.response,
        ]

        result = gemini.recommend("system", "prompt")

        self.assertEqual(result[0]["title"], "Dune")
        self.assertEqual(
            self.called_models(mock_client),
            ["first", "second", "third"],
        )

    @patch("app.providers.gemini.genai.Client")
    def test_all_models_fail(self, mock_client):
        """Raise the last model's error when every model fails."""
        mock_client.return_value.models.generate_content.side_effect = [
            errors.APIError(429, {"error": {"message": "quota"}}),
            errors.APIError(500, {"error": {"message": "internal"}}),
            errors.APIError(503, {"error": {"message": "unavailable"}}),
        ]

        with self.assertRaises(services.ProviderAPIError) as context:
            gemini.recommend("system", "prompt")

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("third", str(context.exception))

    @patch("app.providers.gemini.genai.Client")
    def test_auth_error_stops_fallback(self, mock_client):
        """Don't try other models when the API key is rejected."""
        mock_client.return_value.models.generate_content.side_effect = errors.APIError(
            403, {"error": {"message": "forbidden"}}
        )

        with self.assertRaises(services.ProviderAPIError):
            gemini.recommend("system", "prompt")

        self.assertEqual(self.called_models(mock_client), ["first"])

    @override_settings(GEMINI_MODELS=[])
    def test_no_models(self):
        """Raise when no model is configured."""
        with self.assertRaises(services.ProviderAPIError):
            gemini.recommend("system", "prompt")

    @patch("app.providers.gemini.genai.Client")
    def test_failed_model_skipped_next_time(self, mock_client):
        """Try a recently failed model only after the others."""
        mock_client.return_value.models.generate_content.side_effect = [
            httpx.ReadTimeout("timed out"),
            self.response,
            self.response,
        ]

        gemini.recommend("system", "prompt")
        gemini.recommend("system", "prompt")

        self.assertEqual(
            self.called_models(mock_client),
            ["first", "second", "second"],
        )

    @patch("app.providers.gemini.genai.Client")
    def test_all_models_unavailable_keeps_order(self, mock_client):
        """Try every model in order when all recently failed."""
        for model in ("first", "second", "third"):
            cache.set(gemini.unavailable_cache_key(model), 1)
        mock_client.return_value.models.generate_content.side_effect = [
            httpx.ReadTimeout("timed out"),
            self.response,
        ]

        gemini.recommend("system", "prompt")

        self.assertEqual(self.called_models(mock_client), ["first", "second"])

    @patch("app.providers.gemini.genai.Client")
    def test_auth_error_does_not_skip_model(self, mock_client):
        """Don't mark a model as unavailable because of the API key."""
        mock_client.return_value.models.generate_content.side_effect = errors.APIError(
            401, {"error": {"message": "unauthorized"}}
        )

        with self.assertRaises(services.ProviderAPIError):
            gemini.recommend("system", "prompt")

        self.assertIsNone(cache.get(gemini.unavailable_cache_key("first")))
