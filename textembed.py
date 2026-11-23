"""Text embeddings of song descriptions (a third, semantic view of each song).

If `sentence-transformers` is installed we use a pretrained language model
(all-MiniLM-L6-v2), which knows that e.g. "melancholic" and "sad" are close.
If it is missing or can't load (no internet on first run), we fall back to
TF-IDF + SVD (latent semantic analysis) so the app still works.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def describe_song(s):
    """Turn a song's tags and audio numbers into a short natural-language description."""
    genres = ", ".join(g.replace("_", " ") for g in s["genres"])
    moods = ", ".join(s["moods"])
    bits = [f"{genres} music", f"{moods} mood"]
    bits.append("high energy" if s["energy"] > .65 else "low energy" if s["energy"] < .45
                else "medium energy")
    if s["dance"] > .72:
        bits.append("very danceable")
    if s["acoustic"] > .4:
        bits.append("acoustic")
    bits.append("fast tempo" if s["tempo"] > 125 else "slow tempo" if s["tempo"] < 95
                else "mid tempo")
    return ", ".join(bits)


class TextEmbedder:
    def __init__(self, prefer_pretrained=True):
        self.prefer_pretrained = prefer_pretrained
        self.backend = None

    def fit(self, texts):
        if self.prefer_pretrained:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._st = SentenceTransformer(MODEL_NAME)
                self.backend = "sentence-transformers (all-MiniLM-L6-v2)"
                return self
            except Exception:
                pass  # not installed / offline: use the fallback below
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit(texts)
        n_comp = max(2, min(24, len(texts) - 1, len(self.tfidf.vocabulary_) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(self.tfidf.transform(texts))
        self.backend = "tfidf-svd (install sentence-transformers for the pretrained model)"
        return self

    def encode(self, texts):
        if hasattr(self, "_st"):
            vecs = self._st.encode(list(texts), normalize_embeddings=True)
        else:
            vecs = self.svd.transform(self.tfidf.transform(texts))
        return normalize(np.asarray(vecs)).astype(np.float32)
