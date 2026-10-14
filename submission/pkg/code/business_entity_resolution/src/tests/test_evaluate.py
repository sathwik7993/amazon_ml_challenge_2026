import pytest

from entity_resolution.evaluate import evaluate


def test_problem_statement_example_scores_0_714():
    truth = {"S1-00001": ["S2-00047", "S3-00812"]}
    predicted = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
    assert evaluate(predicted, truth) == pytest.approx(0.714, abs=1e-3)


def test_singleton_scores_one_only_when_predicted_empty():
    truth = {"S1-1": [], "S1-2": []}
    predicted = {"S1-1": [], "S1-2": ["S2-9"]}
    assert evaluate(predicted, truth) == pytest.approx(0.5)


def test_leaderboard_score_is_mean_of_entity_scores_and_missing_prediction_is_empty():
    truth = {"S1-1": ["S2-1"], "S1-2": ["S2-2", "S3-2"], "S1-3": []}
    predicted = {"S1-1": ["S2-1"], "S1-2": ["S3-2"]}
    # entity scores: 1.0, P=1 R=0.5 -> 1.25*0.5/(0.25+0.5)=0.8333, singleton empty -> 1.0
    assert evaluate(predicted, truth) == pytest.approx((1.0 + 0.83333 + 1.0) / 3, abs=1e-4)
