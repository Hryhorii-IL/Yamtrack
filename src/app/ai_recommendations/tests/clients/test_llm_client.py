from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.clients.llm_client import LLMClient
from app.ai_recommendations.bl.errors import LLMClientAuthError, LLMClientError
from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

DUNE = RecommendedTitle(title="Dune")


class ScriptedClient(LLMClient):
    """Plays a script per model: titles to yield, then an optional error."""

    provider = "scripted"
    label = "Scripted"
    api_key_setting = "SCRIPTED_API_KEY"

    def __init__(self, script, models=("first", "second", "third")):
        """Keep the script of each model."""
        super().__init__(models=models, timeout=10)
        self.script = script
        self.called = []

    def generate(self, model, system_instruction, prompt, thinking):  # noqa: ARG002
        """Yield the model's scripted titles, then raise its scripted error."""
        self.called.append(model)
        titles, status_code = self.script.get(model, ([DUNE], None))
        yield from titles
        if status_code is not None:
            raise self.error(status_code, f"{model} failed")


class LLMClientTests(TestCase):
    """Test the model fallback shared by all LLM clients."""

    def setUp(self):
        """Clear models marked as unavailable."""
        cache.clear()

    def test_first_model(self):
        """Use the first model when it works."""
        client = ScriptedClient({})

        self.assertEqual(list(client.recommend("system", "prompt")), [DUNE])
        self.assertEqual(client.called, ["first"])

    def test_fallback_to_next_model(self):
        """Try the next model when one fails and try it last next time."""
        client = ScriptedClient({"first": ([], 500)})

        self.assertEqual(list(client.recommend("system", "prompt")), [DUNE])
        list(client.recommend("system", "prompt"))

        self.assertEqual(client.called, ["first", "second", "second"])

    def test_empty_response_falls_back(self):
        """Treat a model that yields nothing as failed."""
        client = ScriptedClient({"first": ([], None)})

        self.assertEqual(list(client.recommend("system", "prompt")), [DUNE])
        self.assertEqual(client.called, ["first", "second"])

    def test_all_models_fail(self):
        """Raise the last model's error when every model fails."""
        client = ScriptedClient(
            {"first": ([], 429), "second": ([], 500), "third": ([], 503)},
        )

        with self.assertRaises(LLMClientError) as context:
            list(client.recommend("system", "prompt"))

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("Scripted", str(context.exception))

    def test_auth_error_stops_fallback(self):
        """Don't try other models nor skip the model when the key is rejected."""
        client = ScriptedClient({"first": ([], 401)})

        with self.assertRaises(LLMClientAuthError):
            list(client.recommend("system", "prompt"))

        self.assertEqual(client.called, ["first"])
        self.assertIsNone(cache.get(client.unavailable_cache_key("first")))

    def test_failure_mid_stream_keeps_titles(self):
        """Keep the titles yielded before a failure and don't fall back."""
        client = ScriptedClient({"first": ([DUNE], 500)})

        self.assertEqual(list(client.recommend("system", "prompt")), [DUNE])
        self.assertEqual(client.called, ["first"])
        self.assertIsNotNone(cache.get(client.unavailable_cache_key("first")))

    def test_all_unavailable_keeps_order(self):
        """Try every model in order when all recently failed."""
        client = ScriptedClient({"first": ([], 500)})
        for model in client.models:
            cache.set(client.unavailable_cache_key(model), 1)

        list(client.recommend("system", "prompt"))

        self.assertEqual(client.called, ["first", "second"])

    def test_no_models(self):
        """Raise when no model is configured."""
        with self.assertRaises(LLMClientError):
            list(ScriptedClient({}, models=()).recommend("system", "prompt"))

    def test_max_duration(self):
        """Cover every configured model timing out."""
        self.assertEqual(ScriptedClient({}).max_duration(), 30)
