from unittest import TestCase

from app.ai_recommendations.bl.calculations.pool import (
    is_growing,
    needs_extension,
    remaining_slots,
    with_batch,
    with_item,
    without_media,
)
from app.ai_recommendations.bl.types.constants import (
    MAX_EXTENSIONS,
    POOL_SHOWN,
    POOL_SIZE,
)
from app.ai_recommendations.bl.types.pool import RecommendationPool
from app.ai_recommendations.tests.fakes import NOW, resolved


def pool(count=0, **kwargs):
    """Return a pool with ``count`` items."""
    items = tuple(resolved(str(media_id)) for media_id in range(count))
    return RecommendationPool(generated_at=NOW, items=items, **kwargs)


class NeedsExtensionTests(TestCase):
    """Test when a pool gets another batch."""

    def test_small_finished_pool(self):
        """Extend a finished pool below its size."""
        self.assertTrue(needs_extension(pool(1)))

    def test_done(self):
        """Don't extend full, partial, failed or exhausted pools."""
        self.assertFalse(needs_extension(pool(POOL_SIZE)))
        self.assertFalse(needs_extension(pool(1, partial=True)))
        self.assertFalse(needs_extension(pool(1, error="boom")))
        self.assertFalse(needs_extension(pool(1, extensions=MAX_EXTENSIONS)))


class PoolChangeTests(TestCase):
    """Test building new pools from batches."""

    def test_with_item(self):
        """Append one item."""
        self.assertEqual(with_item(pool(1), resolved("9")).items[-1], resolved("9"))

    def test_with_batch(self):
        """Append new items only and count the extension."""
        extended = with_batch(pool(1), [resolved("0"), resolved("5")])

        self.assertEqual(extended.items, (resolved("0"), resolved("5")))
        self.assertEqual(extended.extensions, 1)

    def test_without_media(self):
        """Drop the viewed media from its own recommendations."""
        self.assertEqual(
            without_media([resolved("1"), resolved("2")], "tmdb", "1"),
            (resolved("2"),),
        )


class PageTests(TestCase):
    """Test how many cards a page shows and whether it keeps polling."""

    def test_remaining_slots(self):
        """Fill the page up to POOL_SHOWN cards."""
        self.assertEqual(remaining_slots(POOL_SHOWN - 2), 2)
        self.assertEqual(remaining_slots(POOL_SHOWN + 1), 0)

    def test_growing(self):
        """Poll while the pool is generated or a queued batch may add cards."""
        self.assertTrue(
            is_growing(pool(1, partial=True), shown=1, background_queued=False)
        )
        self.assertTrue(is_growing(pool(1), shown=1, background_queued=True))
        self.assertFalse(is_growing(pool(1), shown=1, background_queued=False))
        self.assertFalse(
            is_growing(pool(1, partial=True), shown=POOL_SHOWN, background_queued=True),
        )
