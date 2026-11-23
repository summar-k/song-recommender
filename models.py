"""The ML parts.

scikit-learn: TF-IDF (tags -> numbers), MinMaxScaler (audio features -> 0..1),
              KMeans (group songs into "style clusters").
PyTorch:      a small denoising autoencoder that learns a compact "style embedding"
              for every song, so similarity is measured in a learned space.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize
from torch import nn

from config import FAMILIES_OF, NUM_WEIGHT, NUMERIC


class FeatureBuilder:
    """Song dicts -> one numeric vector (tag TF-IDF + scaled audio features)."""

    @staticmethod
    def _doc(s):
        fams = [f for g in s["genres"] for f in FAMILIES_OF.get(g, [])]
        return " ".join(s["genres"] * 2 + fams * 2 + s["moods"])  # genre signal > mood signal

    def fit(self, songs):
        self.tfidf = TfidfVectorizer(token_pattern=r"[^ ]+").fit([self._doc(s) for s in songs])
        self.scaler = MinMaxScaler().fit([[s[k] for k in NUMERIC] for s in songs])
        return self

    def transform(self, songs):
        text = normalize(self.tfidf.transform([self._doc(s) for s in songs]).toarray())
        num = self.scaler.transform([[s[k] for k in NUMERIC] for s in songs])
        return normalize(np.hstack([text, NUM_WEIGHT * num])).astype(np.float32)


class StyleAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden=32, d_emb=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(),
                                     nn.Linear(d_hidden, d_emb))
        self.decoder = nn.Sequential(nn.Linear(d_emb, d_hidden), nn.ReLU(),
                                     nn.Linear(d_hidden, d_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_embeddings(X, epochs=400, lr=1e-2, noise=0.1, seed=0):
    """Train the autoencoder; return (L2-normalised embeddings, final loss).

    Denoising: we randomly zero some inputs and ask the net to rebuild the clean
    vector, which forces the 12-number embedding to capture the *essence* of a song's
    style rather than memorise exact values."""
    torch.manual_seed(seed)
    x = torch.from_numpy(X)
    model = StyleAutoencoder(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        corrupted = x * (torch.rand_like(x) > noise)
        loss = loss_fn(model(corrupted), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        emb = model.encoder(x).numpy()
    return normalize(emb), loss.item()


def cluster_styles(Z, songs, k=6, seed=0):
    """KMeans over song vectors; label each cluster by its most common genres."""
    k = min(k, len(songs))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    names = {}
    for c in range(k):
        counts = {}
        for s, l in zip(songs, labels):
            if l == c:
                for g in s["genres"]:
                    counts[g] = counts.get(g, 0) + 1
        top = sorted(counts, key=counts.get, reverse=True)[:2]
        names[c] = " / ".join(top) or f"cluster {c}"
    return labels, names
