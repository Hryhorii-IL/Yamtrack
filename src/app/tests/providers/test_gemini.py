import json
from unittest.mock import MagicMock, patch

import httpx
from django.conf import settings
from django.core.cache import cache
from django.test import TestCase, override_settings
from google.genai import errors

from app.providers import gemini, llm, services

DUNE = {"title": "Dune", "year": 2021}


def stream(*texts):
    """Return streamed response chunks with the given texts."""
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.text = text
        chunks.append(chunk)
    return iter(chunks)


def response_text(*items):
    """Return the JSON text of a RecommendationList."""
    return json.dumps({"items": list(items)})


def failing_stream(error, *texts):
    """Return a stream that yields the given texts and then raises."""
    yield from stream(*texts)
    raise error


class GeminiTests(TestCase):
    """Test the Gemini client wrapper."""

    def setUp(self):
        """Clear models marked as unavailable."""
        cache.clear()

    @patch("app.providers.gemini.genai.Client")
    def test_recommend(self, mock_client):
        """Yield the streamed recommendations as dicts."""
        text = response_text(DUNE, {"title": "Arrival", "year": None})
        mock_client.return_value.models.generate_content_stream.return_value = stream(
            text[:20],
            text[20:50],
            text[50:],
        )

        result = list(gemini.recommend("system", "prompt"))

        self.assertEqual(
            result,
            [DUNE, {"title": "Arrival", "year": None}],
        )
        mock_stream = mock_client.return_value.models.generate_content_stream
        kwargs = mock_stream.call_args.kwargs
        self.assertEqual(kwargs["model"], settings.GEMINI_MODELS[0])
        self.assertEqual(kwargs["contents"], "prompt")
        self.assertEqual(kwargs["config"].system_instruction, "system")

    @patch("app.providers.gemini.genai.Client")
    def test_recommend_api_error(self, mock_client):
        """Wrap Gemini API errors in ProviderAPIError."""
        mock_client.return_value.models.generate_content_stream.side_effect = (
            errors.APIError(429, {"error": {"message": "quota exceeded"}})
        )

        with self.assertRaises(services.ProviderAPIError) as context:
            list(gemini.recommend("system", "prompt"))

        self.assertEqual(context.exception.status_code, 429)
        self.assertIn("Gemini", str(context.exception))

    @patch("app.providers.gemini.genai.Client")
    def test_recommend_invalid_response(self, mock_client):
        """Raise ProviderAPIError when no recommendation can be parsed."""
        mock_client.return_value.models.generate_content_stream.return_value = stream(
            "not json",
        )

        with self.assertRaises(services.ProviderAPIError):
            list(gemini.recommend("system", "prompt"))

    @patch("app.providers.gemini.genai.Client")
    def test_recommend_network_error(self, mock_client):
        """Wrap network errors and timeouts in ProviderAPIError."""
        mock_client.return_value.models.generate_content_stream.side_effect = (
            httpx.ConnectTimeout("timed out")
        )

        with self.assertRaises(services.ProviderAPIError) as context:
            list(gemini.recommend("system", "prompt"))

        self.assertIsNone(context.exception.status_code)

    @patch("app.providers.gemini.genai.Client")
    def test_error_mid_stream_keeps_partial(self, mock_client):
        """Keep the recommendations streamed before a failure."""
        text = response_text(DUNE, DUNE)
        first_item_end = text.index("}") + 1
        mock_client.return_value.models.generate_content_stream.return_value = (
            failing_stream(httpx.ReadTimeout("timed out"), text[:first_item_end])
        )

        result = list(gemini.recommend("system", "prompt"))

        self.assertEqual(result, [DUNE])


@override_settings(GEMINI_MODELS=["first", "second", "third"])
class GeminiFallbackTests(TestCase):
    """Test falling back to the next configured Gemini model."""

    def setUp(self):
        """Clear unavailable models."""
        cache.clear()

    def ok(self):
        """Return a successful stream."""
        return stream(response_text(DUNE))

    def called_models(self, mock_client):
        """Return the models passed to generate_content_stream in order."""
        return [
            call.kwargs["model"]
            for call in (
                mock_client.return_value.models.generate_content_stream.call_args_list
            )
        ]

    @patch("app.providers.gemini.genai.Client")
    def test_fallback_to_next_model(self, mock_client):
        """Try the next model when one fails."""
        mock_client.return_value.models.generate_content_stream.side_effect = [
            errors.APIError(400, {"error": {"message": "bad model name"}}),
            httpx.ReadTimeout("timed out"),
            self.ok(),
        ]

        result = list(gemini.recommend("system", "prompt"))

        self.assertEqual(result, [DUNE])
        self.assertEqual(
            self.called_models(mock_client),
            ["first", "second", "third"],
        )

    @patch("app.providers.gemini.genai.Client")
    def test_fallback_after_empty_stream(self, mock_client):
        """Try the next model when a stream fails before any recommendation."""
        mock_client.return_value.models.generate_content_stream.side_effect = [
            failing_stream(httpx.ReadTimeout("timed out"), '{"items": ['),
            self.ok(),
        ]

        result = list(gemini.recommend("system", "prompt"))

        self.assertEqual(result, [DUNE])
        self.assertEqual(self.called_models(mock_client), ["first", "second"])

    @patch("app.providers.gemini.genai.Client")
    def test_no_fallback_after_partial_stream(self, mock_client):
        """Don't mix models once recommendations were yielded."""
        text = response_text(DUNE, DUNE)
        mock_client.return_value.models.generate_content_stream.side_effect = [
            failing_stream(httpx.ReadTimeout("timed out"), text[: text.index("}") + 1]),
            self.ok(),
        ]

        result = list(gemini.recommend("system", "prompt"))

        self.assertEqual(result, [DUNE])
        self.assertEqual(self.called_models(mock_client), ["first"])
        self.assertIsNotNone(cache.get(llm.unavailable_cache_key("gemini", "first")))

    @patch("app.providers.gemini.genai.Client")
    def test_all_models_fail(self, mock_client):
        """Raise the last model's error when every model fails."""
        mock_client.return_value.models.generate_content_stream.side_effect = [
            errors.APIError(429, {"error": {"message": "quota"}}),
            errors.APIError(500, {"error": {"message": "internal"}}),
            errors.APIError(503, {"error": {"message": "unavailable"}}),
        ]

        with self.assertRaises(services.ProviderAPIError) as context:
            list(gemini.recommend("system", "prompt"))

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("third", str(context.exception))

    @patch("app.providers.gemini.genai.Client")
    def test_auth_error_stops_fallback(self, mock_client):
        """Don't try other models when the API key is rejected."""
        mock_client.return_value.models.generate_content_stream.side_effect = (
            errors.APIError(403, {"error": {"message": "forbidden"}})
        )

        with self.assertRaises(services.ProviderAPIError):
            list(gemini.recommend("system", "prompt"))

        self.assertEqual(self.called_models(mock_client), ["first"])

    @override_settings(GEMINI_MODELS=[])
    def test_no_models(self):
        """Raise when no model is configured."""
        with self.assertRaises(services.ProviderAPIError):
            list(gemini.recommend("system", "prompt"))

    @patch("app.providers.gemini.genai.Client")
    def test_failed_model_skipped_next_time(self, mock_client):
        """Try a recently failed model only after the others."""
        mock_client.return_value.models.generate_content_stream.side_effect = [
            httpx.ReadTimeout("timed out"),
            self.ok(),
            self.ok(),
        ]

        list(gemini.recommend("system", "prompt"))
        list(gemini.recommend("system", "prompt"))

        self.assertEqual(
            self.called_models(mock_client),
            ["first", "second", "second"],
        )

    @patch("app.providers.gemini.genai.Client")
    def test_all_models_unavailable_keeps_order(self, mock_client):
        """Try every model in order when all recently failed."""
        for model in ("first", "second", "third"):
            cache.set(llm.unavailable_cache_key("gemini", model), 1)
        mock_client.return_value.models.generate_content_stream.side_effect = [
            httpx.ReadTimeout("timed out"),
            self.ok(),
        ]

        list(gemini.recommend("system", "prompt"))

        self.assertEqual(self.called_models(mock_client), ["first", "second"])

    @patch("app.providers.gemini.genai.Client")
    def test_auth_error_does_not_skip_model(self, mock_client):
        """Don't mark a model as unavailable because of the API key."""
        mock_client.return_value.models.generate_content_stream.side_effect = (
            errors.APIError(401, {"error": {"message": "unauthorized"}})
        )

        with self.assertRaises(services.ProviderAPIError):
            list(gemini.recommend("system", "prompt"))

        self.assertIsNone(cache.get(llm.unavailable_cache_key("gemini", "first")))
