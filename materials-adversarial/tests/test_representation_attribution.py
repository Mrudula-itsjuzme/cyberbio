import numpy as np
import pytest

from materials_adv.evaluation.representation_attribution import (
    RidgeRegressor, length_features, paired_bootstrap_ci, representation_invariance_score,
    representation_statistics, safe_correlation, token_features,
)


def test_invariance_score_is_one_for_identical_predictions_and_decreases():
    assert representation_invariance_score([2, 2, 2], 1.0) == 1.0
    assert representation_invariance_score([1, 2, 3], 1.0) < 1.0
    with pytest.raises(ValueError):
        representation_invariance_score([], 1.0)


def test_representation_statistics_exact_scalar_quantities():
    result = representation_statistics([1.0, 2.0, 4.0], [2, 3, 4],
                                       np.array([[0, 0], [3, 4], [0, 0]]),
                                       reference_scale=2.0)
    assert result["prediction_mean"] == pytest.approx(7 / 3)
    assert result["prediction_range"] == 3.0
    assert result["max_pairwise_prediction_drift"] == 3.0
    assert result["length_prediction_correlation"] == pytest.approx(0.9819805)
    assert result["embedding_max_pairwise_distance"] == 5.0


def test_token_count_and_frequency_features_separate_length():
    strings, vocab = ["CC", "CCCC"], ["C"]
    assert token_features(strings, vocab, frequencies=False).ravel().tolist() == [2, 4]
    assert token_features(strings, vocab, frequencies=True).ravel().tolist() == [1, 1]
    assert length_features(strings).ravel().tolist() == [2, 4]


def test_ridge_fits_simple_linear_relationship():
    x = np.arange(5, dtype=float)[:, None]
    model = RidgeRegressor(alpha=0.0).fit(x, 2 * x.ravel() + 1)
    assert model.predict(np.array([[5.0]]))[0] == pytest.approx(11.0)


def test_correlation_and_bootstrap_insufficient_sample_are_explicit():
    assert safe_correlation([1, 1, 1], [1, 2, 3]) is None
    result = paired_bootstrap_ci([0.1, 0.2], rng=np.random.default_rng(1), minimum_n=3)
    assert result["status"] == "insufficient_sample"
    assert result["ci_low"] is None


def test_paired_bootstrap_is_reproducible_and_contains_constant_mean():
    result = paired_bootstrap_ci([0.5] * 20, rng=np.random.default_rng(3), n_resamples=100)
    assert result["status"] == "estimated"
    assert result["ci_low"] == pytest.approx(0.5)
    assert result["ci_high"] == pytest.approx(0.5)


def test_transformer_encode_matches_forward_head():
    torch = pytest.importorskip("torch")
    from materials_adv.models.transformer import TransformerRegressorModel
    model = TransformerRegressorModel(4, 8, 1, 2, 16, 0.0, 10, "mean").eval()
    ids = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[False, False, True]])
    with torch.no_grad():
        encoded = model.encode(ids, mask)
        assert torch.allclose(model(ids, mask), model.regressor(encoded).squeeze(-1))
    assert encoded.shape == (1, 8)
