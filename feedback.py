"""Learn from likes and dislikes (PyTorch).

A tiny logistic-regression model is trained on the fly:
  positives  = songs you liked (strong) + songs by the artists you searched (weak)
  negatives  = songs you disliked (strong) + a few random unrated songs (very weak)
It then scores every song with "probability you'd like it", which nudges the ranking.
"""
import numpy as np
import torch
from torch import nn


class PreferenceModel:
    def __init__(self, epochs=300, lr=0.05, weight_decay=1e-2, seed=0):
        self.epochs, self.lr, self.wd, self.seed = epochs, lr, weight_decay, seed

    def fit(self, Z, seeds, liked, disliked):
        """Z: (n_songs, d) features. seeds/liked/disliked: lists of row indices."""
        rng = np.random.default_rng(self.seed)
        rows, labels, weights = [], [], []
        for idx, y, w in ((seeds, 1, 0.3), (liked, 1, 1.0), (disliked, 0, 1.0)):
            for i in idx:
                rows.append(i)
                labels.append(y)
                weights.append(w)
        rated = set(seeds) | set(liked) | set(disliked)
        pool = [i for i in range(len(Z)) if i not in rated]
        n_pos = sum(labels)
        if pool and n_pos > len(disliked):          # add weak pseudo-negatives
            extra = rng.choice(pool, size=min(len(pool), 2 * n_pos), replace=False)
            rows += list(extra)
            labels += [0] * len(extra)
            weights += [0.1] * len(extra)

        torch.manual_seed(self.seed)
        x = torch.tensor(Z[rows], dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        w = torch.tensor(weights, dtype=torch.float32)
        model = nn.Linear(x.shape[1], 1)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        for _ in range(self.epochs):
            loss = (nn.functional.binary_cross_entropy_with_logits(
                model(x).squeeze(1), y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.model = model
        return self

    def score(self, Z):
        """Standardised preference score per song (0 = average, +1 = one std above)."""
        with torch.no_grad():
            logits = self.model(torch.tensor(Z, dtype=torch.float32)).squeeze(1).numpy()
        return (logits - logits.mean()) / (logits.std() + 1e-8)
