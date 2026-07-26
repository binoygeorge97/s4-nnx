import pytest

from s4_nnx import S4Regressor


def test_fit_and_predict_returns_mean():
    model = S4Regressor().fit([0, 1, 2], [2.0, 4.0, 6.0])
    assert model.predict([3, 4]) == [4.0, 4.0]


def test_fit_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        S4Regressor().fit([0, 1], [1.0])


def test_predict_raises_before_fit():
    with pytest.raises(RuntimeError, match="fit before predict"):
        S4Regressor().predict([1])
