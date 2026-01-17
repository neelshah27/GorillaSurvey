"""
Tests for the metrics computation module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics_tools import (
    compute_efi,
    compute_ids,
    compute_nps,
    compute_all_metrics,
    compute_deflection_score,
    compute_sentiment_heuristic,
    choose_strategy,
)


class TestDeflectionScore:
    """Tests for deflection detection."""

    def test_detects_idk(self):
        assert compute_deflection_score("idk") == 1.0
        assert compute_deflection_score("I don't know") == 1.0

    def test_detects_busy(self):
        assert compute_deflection_score("busy right now") == 1.0
        assert compute_deflection_score("gtg") == 1.0

    def test_detects_short_responses(self):
        assert compute_deflection_score("ok") == 1.0
        assert compute_deflection_score("k") == 1.0

    def test_normal_response(self):
        assert compute_deflection_score("I really loved the product") == 0.0


class TestSentimentHeuristic:
    """Tests for sentiment detection."""

    def test_positive_sentiment(self):
        score = compute_sentiment_heuristic("I love this! Amazing quality!")
        assert score > 0

    def test_negative_sentiment(self):
        score = compute_sentiment_heuristic("Terrible, hate it, waste of money")
        assert score < 0

    def test_neutral_sentiment(self):
        score = compute_sentiment_heuristic("It arrived yesterday")
        assert score == 0.0


class TestEFI:
    """Tests for Engagement Friction Index."""

    def test_low_friction_engaged_user(self):
        result = compute_efi(
            text="I absolutely loved it! The fabric is so soft and comfortable.",
            latency_seconds=5.0,
            turn=2,
            previous_efi=0.2,
        )
        assert result.current < 0.3  # Should be engaged

    def test_high_friction_disengaged_user(self):
        result = compute_efi(
            text="idk whatever",
            latency_seconds=55.0,
            turn=10,
            previous_efi=0.6,
        )
        assert result.current > 0.5  # Should show friction

    def test_action_thresholds(self):
        # Low EFI -> continue_normal
        result = compute_efi("Great product!", 5.0, 2, 0.1)
        assert result.action == "continue_normal"


class TestIDS:
    """Tests for Insight Depth Score."""

    def test_high_depth_detailed_response(self):
        result = compute_ids(
            "The fabric is super soft cotton, fits perfectly because I sized up. "
            "The stitching is excellent quality."
        )
        assert result.current > 0.5

    def test_low_depth_vague_response(self):
        result = compute_ids("it's fine I guess")
        assert result.current < 0.3
        assert result.should_probe == True

    def test_probing_logic(self):
        # Should probe when low depth and not already probed
        result = compute_ids("ok", field_already_probed=False)
        assert result.should_probe == True

        # Should not probe if already probed
        result = compute_ids("ok", field_already_probed=True)
        assert result.should_probe == False


class TestNPS:
    """Tests for Net Promoter Score inference."""

    def test_promoter_signals(self):
        result = compute_nps(
            "I love it! Definitely recommending to all my friends!",
            previous_logit=0.0,
            turn=3,
        )
        assert result.bucket == "promoter"
        assert result.p_advocacy > 0.7

    def test_detractor_signals(self):
        result = compute_nps(
            "Terrible quality, returning it. Never buying again.",
            previous_logit=0.0,
            turn=3,
        )
        assert result.bucket == "detractor"
        assert result.p_advocacy < 0.4

    def test_cumulative_updates(self):
        # Start neutral
        result1 = compute_nps("It's okay", previous_logit=0.0, turn=1)

        # Add positive signal
        result2 = compute_nps("Actually I love it!", previous_logit=result1.logit, turn=2)

        assert result2.p_advocacy > result1.p_advocacy


class TestStrategy:
    """Tests for strategy selection."""

    def test_exit_on_high_friction(self):
        strategy = choose_strategy(
            efi=0.9,
            ids=0.5,
            fields_remaining=3,
            nps_should_ask=False,
            can_probe=False,
        )
        assert strategy == "exit_graceful"

    def test_quick_reply_on_moderate_friction(self):
        strategy = choose_strategy(
            efi=0.6,
            ids=0.5,
            fields_remaining=3,
            nps_should_ask=False,
            can_probe=False,
        )
        assert strategy == "quick_reply_options"

    def test_probe_on_low_depth(self):
        strategy = choose_strategy(
            efi=0.2,
            ids=0.2,
            fields_remaining=3,
            nps_should_ask=False,
            can_probe=True,
        )
        assert strategy == "empathize_followup"

    def test_continue_normal_default(self):
        strategy = choose_strategy(
            efi=0.2,
            ids=0.5,
            fields_remaining=3,
            nps_should_ask=False,
            can_probe=False,
        )
        assert strategy == "continue_normal"


class TestIntegration:
    """Integration tests for full metrics computation."""

    def test_full_metrics_computation(self):
        metrics = compute_all_metrics(
            text="The hoodie is amazing! Fabric is super soft and fits perfectly.",
            latency_seconds=8.0,
            turn=3,
            previous_efi=0.2,
            previous_nps_logit=0.0,
            session_id="test_session",
        )

        assert metrics.session_id == "test_session"
        assert metrics.turn == 3
        assert 0 <= metrics.efi.current <= 1
        assert 0 <= metrics.ids.current <= 1
        assert 0 <= metrics.nps.p_advocacy <= 1

    def test_metrics_to_dict(self):
        metrics = compute_all_metrics(
            text="Great product!",
            latency_seconds=5.0,
            turn=2,
            session_id="test",
        )

        data = metrics.to_dict()
        assert "efi" in data
        assert "ids" in data
        assert "nps" in data
        assert data["session_id"] == "test"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
