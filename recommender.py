"""Trend-aware song recommender (scikit-learn + PyTorch + text embeddings).

    python recommender.py "Bad Bunny, SZA, Tame Impala"
"""
import argparse
import difflib
import re
from collections import Counter

import numpy as np

import data
from config import (BASE_YEAR, BLOCKED_ARTISTS, DATA_FILE, FEEDBACK_MIN_VOTES, FEEDBACK_WEIGHT,
                    RECENCY_BONUS, RECENT_SPAN, SPACE_WEIGHTS)
from feedback import PreferenceModel
from models import FeatureBuilder, cluster_styles, encode_styles, train_autoencoder
from textembed import TextEmbedder, describe_song


def song_id(s):
    return f"{s['artist']}|{s['title']}"


class SongRecommender:
    def __init__(self, songs=None, use_torch=True, use_text=True, seed=0):
        self.songs = songs if songs is not None else data.load_songs(DATA_FILE)
        self.use_torch, self.use_text, self.seed = use_torch, use_text, seed
        self._fit()

    # ---------- fitting ----------
    def _fit(self):
        self.fb = FeatureBuilder().fit(self.songs)
        X = self.fb.transform(self.songs)                           # scikit-learn view
        self.spaces = {"features": X}
        self.ae = None
        if self.use_torch:
            self.ae, self.ae_loss = train_autoencoder(X, seed=self.seed)    # PyTorch view
            self.spaces["style"] = encode_styles(self.ae, X)
        self.embedder = None
        if self.use_text:
            self.embedder = TextEmbedder().fit([describe_song(s) for s in self.songs])
            self.spaces["text"] = self.embedder.encode([describe_song(s) for s in self.songs])
        self.X = X
        self.labels, self.cluster_names = cluster_styles(
            self.spaces.get("style", X), self.songs, seed=self.seed)
        # only PRIMARY artists are searchable; featured-only names (e.g. a guest rapper) are not
        self.artists = {self._norm(s["artist"]): s["artist"] for s in self.songs}
        gc = Counter(g for s in self.songs for g in s["genres"])
        self.genres = {self._gkey(g): g for g, c in gc.items() if c >= 5}
        self.genres.update({"rb": "rnb", "randb": "rnb", "rnb": "rnb", "hiphop": "hip_hop",
                            "dance": "dance_pop", "electronic": "electronic", "latin": "latin_pop",
                            "alternative": "alt_pop", "alt": "alt_pop", "indie": "indie"})
        self.ids = {song_id(s): i for i, s in enumerate(self.songs)}
        tw = sum(SPACE_WEIGHTS[k] for k in self.spaces)
        self.weights = {k: SPACE_WEIGHTS[k] / tw for k in self.spaces}

    @property
    def text_backend(self):
        return self.embedder.backend if self.embedder else None

    # ---------- artists ----------
    @staticmethod
    def _norm(name):
        """'A$AP Rocky' and 'asap rocky' should match: drop punctuation, ignore case."""
        return re.sub(r"[^a-z0-9 ]", "", name.casefold().replace("$", "s")).strip()

    @staticmethod
    def _gkey(name):
        return re.sub(r"[^a-z0-9]", "", name.casefold())

    @staticmethod
    def genre_label(tag):
        return {"rnb": "r&b", "alt_rnb": "alt r&b"}.get(tag, tag.replace("_", " "))

    def genre_names(self):
        return sorted({self.genre_label(g) for g in self.genres.values()})

    def find_genre(self, name):
        key = self._gkey(name)
        if key in self.genres:
            return self.genres[key]
        close = difflib.get_close_matches(key, self.genres, n=1, cutoff=0.85)
        return self.genres[close[0]] if close else None

    def find_artist(self, name):
        key = self._norm(name)
        if key in self.artists:
            return self.artists[key]
        close = difflib.get_close_matches(key, self.artists, n=1, cutoff=0.75)
        if close:
            return self.artists[close[0]]
        # partial names like "kanye" or "ye" -> a single artist whose name has that word
        if len(key) >= 2:
            hits = [k for k in self.artists if key in k.split()]
            if len(hits) == 1:
                return self.artists[hits[0]]
        return None

    def artist_names(self):
        return sorted(set(self.artists.values()), key=str.casefold)

    # ---------- profile ----------
    def describe(self, rows, idx):
        top = lambda key, k: [t for t, _ in Counter(t for r in rows for t in r[key]).most_common(k)]
        mean = lambda k: float(np.mean([r[k] for r in rows]))
        traits = ["high-energy" if mean("energy") > .65 else "mellow" if mean("energy") < .45
                  else "moderate-energy",
                  "upbeat" if mean("valence") > .6 else "emotional/moody" if mean("valence") < .45
                  else "balanced mood",
                  "danceable" if mean("dance") > .72 else None,
                  "acoustic" if mean("acoustic") > .4 else None]
        styles = [self.cluster_names[c] for c, _ in
                  Counter(self.labels[i] for i in idx).most_common(2)]
        return {"genres": top("genres", 4), "moods": top("moods", 4),
                "traits": [t for t in traits if t], "avg_tempo": round(mean("tempo")),
                "style_clusters": styles}

    # ---------- recommend ----------
    def _similarity(self, seed_vecs):
        """Weighted cosine to the taste centroid, summed over every embedding space."""
        sims = np.zeros(len(self.songs))
        for k, M in self.spaces.items():
            c = seed_vecs[k].mean(axis=0)
            sims += self.weights[k] * (M @ (c / np.linalg.norm(c)))
        return sims

    def recommend(self, artists, n=10, per_artist=2, liked=(), disliked=()):
        found, genres_found, missing, blocked = [], [], [], []
        for a in artists:
            if a.strip().casefold() in BLOCKED_ARTISTS:
                blocked.append(a.strip())
                continue
            exact = self._norm(a) in self.artists
            hit = self.find_artist(a) if exact else None
            gen = None if hit else self.find_genre(a)
            if not hit and not gen:
                hit = self.find_artist(a)
            if hit:
                found.append(hit)
            elif gen:
                genres_found.append(gen)
            else:
                missing.append(a.strip())
        # a taste "group" is either an artist or a genre; each is a set of seed songs
        group_rows = {a: [i for i, s in enumerate(self.songs) if a in s["artists"]] for a in found}
        for g in genres_found:
            group_rows[self.genre_label(g)] = [i for i, s in enumerate(self.songs) if g in s["genres"]]
        idx = sorted({i for r in group_rows.values() for i in r})
        out = {"found": found + [self.genre_label(g) for g in genres_found], "missing": missing, "blocked": blocked,
               "profile": None, "songs": [], "feedback_used": False}
        if not idx:
            return out

        # per-artist taste vectors (used for the combined taste AND for "closest artist")
        groups = {a: {k: M[rows] for k, M in self.spaces.items()}
                  for a, rows in group_rows.items() if rows}
        # one centroid per artist/genre so a big genre can't drown out a single artist
        seed_vecs = {k: np.vstack([g[k].mean(axis=0) for g in groups.values()]) for k in self.spaces}

        sims = self._similarity(seed_vecs)
        years = np.array([max(s["year"], BASE_YEAR) for s in self.songs])
        sims = sims + RECENCY_BONUS * np.clip(years - BASE_YEAR, 0, RECENT_SPAN) / RECENT_SPAN

        # learn from likes / dislikes
        like_i = [self.ids[x] for x in liked if x in self.ids]
        dis_i = [self.ids[x] for x in disliked if x in self.ids]
        if len(like_i) + len(dis_i) >= FEEDBACK_MIN_VOTES:
            Z = np.hstack([np.sqrt(self.weights[k]) * M for k, M in self.spaces.items()])
            pref = PreferenceModel(seed=self.seed).fit(Z, idx, like_i, dis_i)
            sims = sims + FEEDBACK_WEIGHT * pref.score(Z)
            out["feedback_used"] = True

        profile_rows = [self.songs[i] for i in idx]
        profile = out["profile"] = self.describe(profile_rows, idx)
        tags = set(profile["genres"] + profile["moods"])
        skip = set(dis_i)
        per, picks = {}, []
        for i in np.argsort(-sims):
            s = self.songs[i]
            if i in skip or set(s["artists"]) & set(found):
                continue
            if per.get(s["artist"], 0) >= per_artist:
                continue
            per[s["artist"]] = per.get(s["artist"], 0) + 1
            closest, best = None, -1.0
            for a, g in groups.items():              # which of YOUR artists is it nearest to?
                c = sum(self.weights[k] * float(np.mean(g[k] @ self.spaces[k][i]))
                        for k in self.spaces)
                if c > best:
                    closest, best = a, c
            because = [t for t in s["genres"] + s["moods"] if t in tags][:3]
            why = (f"Sounds closest to {closest}" +
                   (f" - shares {', '.join(t.replace('_', ' ') for t in because)}." if because else "."))
            picks.append({"id": song_id(s), "artist": " & ".join(s["artists"]),
                          "title": s["title"], "year": s["year"],
                          "score": round(float(sims[i]), 2),
                          "style": self.cluster_names[self.labels[i]], "because": because,
                          "closest": closest, "why": why})
            if len(picks) == n:
                break
        out["songs"] = picks
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artists", nargs="*", help='e.g. "Bad Bunny, SZA"')
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--no-torch", action="store_true", help="skip the PyTorch autoencoder")
    ap.add_argument("--no-text", action="store_true", help="skip the text embeddings")
    args = ap.parse_args()
    names = [a for a in " ".join(args.artists).split(",") if a.strip()] or \
        [a for a in input("Artists you like (comma-separated): ").split(",") if a.strip()]

    rec = SongRecommender(use_torch=not args.no_torch, use_text=not args.no_text)
    r = rec.recommend(names, n=args.n)
    if r["blocked"]:
        print("Skipped (blocked in config.py):", ", ".join(r["blocked"]))
    if r["missing"]:
        print("Not recognized:", ", ".join(r["missing"]))
    if not r["profile"]:
        return
    p = r["profile"]
    print(f"\nYour taste: {', '.join(p['genres'])} | moods: {', '.join(p['moods'])}")
    print(f"Vibe: {', '.join(p['traits'])} (~{p['avg_tempo']} BPM)")
    print(f"Style clusters: {' + '.join(p['style_clusters'])}\n")
    for i, s in enumerate(r["songs"], 1):
        print(f"{i:>2}. {s['title']} - {s['artist']} ({s['year']}) {s['score']:.0%}")
        print(f"    {s['why']}")


if __name__ == "__main__":
    main()
