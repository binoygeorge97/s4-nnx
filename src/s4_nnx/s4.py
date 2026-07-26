"""Minimal S4 module placeholder."""


class S4Regressor:
    """A tiny baseline regressor used for repository scaffolding."""

    def fit(self, x, y):
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if not y:
            raise ValueError("y must not be empty")
        self._mean = sum(y) / len(y)
        return self

    def predict(self, x):
        if not hasattr(self, "_mean"):
            raise RuntimeError("Model must be fit before predict")
        return [self._mean for _ in x]
