from __future__ import annotations

import pydantic


class RecommendationSchema(pydantic.BaseModel):
    """A recommended title in the structured output the LLM must follow."""

    title: str
    year: int | None = None


class RecommendationListSchema(pydantic.BaseModel):
    """The structured output schema sent to the LLM APIs."""

    items: list[RecommendationSchema]
