"""
Tests for the field extraction module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction_tools import (
    extract_overall_satisfaction,
    extract_fit_rating,
    extract_fabric_quality,
    extract_would_recommend,
    extract_improvement_suggestion,
    extract_all_fields,
)


class TestOverallSatisfaction:
    """Tests for overall satisfaction extraction."""

    def test_extracts_loved(self):
        result = extract_overall_satisfaction("I absolutely loved it!")
        assert result.extracted_value == "loved"
        assert result.confidence > 0.5

    def test_extracts_liked(self):
        result = extract_overall_satisfaction("It's pretty good, I liked it")
        assert result.extracted_value == "liked"

    def test_extracts_neutral(self):
        result = extract_overall_satisfaction("It's okay I guess")
        assert result.extracted_value == "neutral"

    def test_extracts_disappointed(self):
        result = extract_overall_satisfaction("Really disappointed with this")
        assert result.extracted_value == "disappointed"

    def test_no_match(self):
        result = extract_overall_satisfaction("It arrived yesterday")
        assert result.extracted_value is None


class TestFitRating:
    """Tests for fit rating extraction."""

    def test_extracts_perfect(self):
        result = extract_fit_rating("The fit was perfect!")
        assert result.extracted_value == "perfect"

    def test_extracts_small(self):
        result = extract_fit_rating("Runs a bit small, had to size up")
        assert result.extracted_value == "slightly_small"

    def test_extracts_large(self):
        result = extract_fit_rating("It's too big, runs large")
        assert result.extracted_value == "slightly_large"

    def test_extracts_wrong_size(self):
        result = extract_fit_rating("Completely wrong size, way too small")
        assert result.extracted_value == "wrong_size"


class TestFabricQuality:
    """Tests for fabric quality extraction."""

    def test_extracts_excellent(self):
        result = extract_fabric_quality("The fabric is amazing, so soft!")
        assert result.extracted_value == "excellent"

    def test_extracts_good(self):
        result = extract_fabric_quality("Good quality material")
        assert result.extracted_value == "good"

    def test_extracts_poor(self):
        result = extract_fabric_quality("Cheap flimsy fabric")
        assert result.extracted_value == "poor"


class TestWouldRecommend:
    """Tests for recommendation intent extraction."""

    def test_extracts_yes(self):
        result = extract_would_recommend("Definitely would recommend!")
        assert result.extracted_value == "yes"

    def test_extracts_no(self):
        result = extract_would_recommend("No, wouldn't recommend")
        assert result.extracted_value == "no"

    def test_extracts_maybe(self):
        result = extract_would_recommend("Maybe, depends on the person")
        assert result.extracted_value == "maybe"


class TestImprovementSuggestion:
    """Tests for improvement suggestion extraction."""

    def test_extracts_suggestion(self):
        result = extract_improvement_suggestion(
            "Would be nice if you had more color options"
        )
        assert result.extracted_value is not None
        assert result.confidence > 0.5

    def test_extracts_none(self):
        result = extract_improvement_suggestion("Nothing comes to mind")
        assert result.extracted_value == "none"


class TestMultiFieldExtraction:
    """Tests for extracting multiple fields from one message."""

    def test_extracts_multiple_fields(self):
        result = extract_all_fields(
            "Loved it! Fabric is super soft and fits perfectly.",
            session_id="test",
            turn=1,
        )

        field_ids = [e.field_id for e in result.extractions]
        assert "overall_satisfaction" in field_ids
        assert "fabric_quality" in field_ids

    def test_tracks_no_match_fields(self):
        result = extract_all_fields(
            "It arrived yesterday",
            session_id="test",
            turn=1,
        )

        assert len(result.no_match_fields) > 0

    def test_comprehensive_response(self):
        result = extract_all_fields(
            "I loved the hoodie! Perfect fit, amazing soft fabric. "
            "Would definitely recommend. Maybe add more colors?",
            session_id="test",
            turn=1,
        )

        field_ids = [e.field_id for e in result.extractions]
        assert "overall_satisfaction" in field_ids
        assert "fit_rating" in field_ids
        assert "fabric_quality" in field_ids
        assert "would_recommend" in field_ids


class TestConfidenceScores:
    """Tests for extraction confidence scoring."""

    def test_high_confidence_clear_signal(self):
        result = extract_overall_satisfaction("I absolutely loved it!")
        assert result.confidence >= 0.6

    def test_captures_quote(self):
        result = extract_overall_satisfaction("The product is amazing!")
        assert len(result.quote) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
