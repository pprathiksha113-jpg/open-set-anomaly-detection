import numpy as np
from sklearn.linear_model import LogisticRegression


class ConformalCalibrator:
    """Open-set calibration from NORMAL scores only.
    p_value = how often normal images score at least this high."""

    def fit(self, normal_scores):
        self.s = np.sort(np.asarray(normal_scores, dtype=float))
        return self

    def p_value(self, score):
        score = np.atleast_1d(score)
        n = len(self.s)
        ge = n - np.searchsorted(self.s, score, side="left")
        return (ge + 1) / (n + 1)

    def anomaly_confidence(self, score):
        return 1.0 - self.p_value(score)

    def threshold(self, alpha):
        """Score above which about alpha of normal images would be flagged."""
        return float(np.quantile(self.s, 1.0 - alpha))


class PlattCalibrator:
    """Needs a few labelled defects (a dev split only) to turn scores into probabilities."""

    def fit(self, scores, labels):
        s = np.asarray(scores, dtype=float)
        self.mu, self.sd = s.mean(), s.std() + 1e-8
        self.lr = LogisticRegression().fit(((s - self.mu) / self.sd).reshape(-1, 1), labels)
        return self

    def predict_proba(self, scores):
        z = ((np.asarray(scores, dtype=float) - self.mu) / self.sd).reshape(-1, 1)
        return self.lr.predict_proba(z)[:, 1]


def ece(probs, labels, n_bins=10):
    """Expected calibration error."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(err)