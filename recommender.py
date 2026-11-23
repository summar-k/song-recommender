"""Trend-aware song recommender (scikit-learn + PyTorch, optional Claude).

    python recommender.py "Bad Bunny, SZA, Tame Impala"
    python recommender.py "Doechii" --explain        # needs ANTHROPIC_API_KEY
"""
import argparse
import difflib
import json
import os
from collections import Counter

import numpy as np

import data
import re

from config import (BASE_YEAR, BLOCKED_ARTISTS, DATA_FILE, NUMERIC, RECENCY_BONUS,
                    RECENT_SPAN)
from models import FeatureBuilder, cluster_styles, train_embeddings

MODEL = "claude-sonnet-5-5"


class SongRecommender:
    def __init__(self, songs=None, client=None, use_torch=True, seed=0):
        self.songs = songs if songs is not None else data.load_songs(DATA_FILE)
        self.client, self.use_torch, self.seed = client, use_torch, seed
        self._fit()

    def _fit(self):
        self.fb = FeatureBuilder().fit(self.songs)
        self.X = self.fb.transform(self.songs)                          # scikit-learn features
        self.E = None
        if self.use_torch:
            self.E, self.ae_loss = train_embeddings(self.X, seed=self.seed)  # PyTorch embeddings
        self.labels, self.cluster_names = cluster_styles(
            self.E if self.E is not None else self.X, self.songs, seed=self.seed)
        self.artists = {self._norm(a): a for s in self.songs for a in s["artists"]}

    # ---------- artists ----------
    @staticmethod
    def _norm(name):
        """'A$AP Rocky' and 'asap rocky' should match: drop punctuation, ignore case."""
        return re.sub(r"[^a-z0-9 ]", "", name.casefold().replace("$", "s")).strip()

    def find_artist(self, name):
        key = self._norm(name)
        if key in self.artists:
            return self.artists[key]
        close = difflib.get_close_matches(key, self.artists, n=1, cutoff=0.75)
        return self.artists[close[0]] if close else None

    def add_artist_with_ai(self, name):
        """Claude infers an unknown artist's style so they can still seed a profile."""
        prompt = (f'Describe the musical style of the artist "{name}". Reply ONLY with JSON: '
                  '{"genres":[3 snake_case genres],"moods":[3 single-word moods],"energy":0-1,'
                  '"valence":0-1,"dance":0-1,"acoustic":0-1,"tempo":bpm}. '
                  'If you do not know the artist reply {"unknown":true}.')
        resp = self.client.messages.create(model=MODEL, max_tokens=300,
                                           messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
        if d.get("unknown"):
            return None
        s = data._make(name, f"(style of {name})", 0, d["genres"], d["moods"],
                          [d[k] for k in NUMERIC])
        s["virtual"] = True
        if not data.is_allowed(s["artists"], s["genres"]):
            return None
        self.songs.append(s)
        self._fit()
        return name

    # ---------- profile ----------
    def describe(self, idx):
        rows = [self.songs[i] for i in idx]
        top = lambda key, k: [t for t, _ in Counter(t for r in rows for t in r[key]).most_common(k)]
        mean = lambda k: float(np.mean([r[k] for r in rows]))
        traits = ["high-energy" if mean("energy") > .65 else "mellow" if mean("energy") < .45
                  else "moderate-energy",
                  "upbeat" if mean("valence") > .6 else "emotional/moody" if mean("valence") < .45
                  else "balanced mood",
                  "danceable" if mean("dance") > .72 else None,
                  "acoustic" if mean("acoustic") > .4 else None]
        styles = [self.cluster_names[c] for c, _ in Counter(self.labels[i] for i in idx).most_common(2)]
        return {"genres": top("genres", 4), "moods": top("moods", 4),
                "traits": [t for t in traits if t], "avg_tempo": round(mean("tempo")),
                "style_clusters": styles}

    # ---------- recommend ----------
    def recommend(self, artists, n=10, per_artist=2):
        found, missing, blocked = [], [], []
        for a in artists:
            if a.strip().casefold() in BLOCKED_ARTISTS:
                blocked.append(a.strip())
                continue
            hit = self.find_artist(a)
            if not hit and self.client:
                try:
                    hit = self.add_artist_with_ai(a)
                except Exception:
                    hit = None
            (found if hit else missing).append(hit or a.strip())
        idx = [i for i, s in enumerate(self.songs) if set(s["artists"]) & set(found)]
        out = {"found": found, "missing": missing, "blocked": blocked, "profile": None, "songs": []}
        if not idx:
            return out

        def cos_to_centroid(M):
            c = M[idx].mean(axis=0)
            return M @ (c / np.linalg.norm(c))

        sims = cos_to_centroid(self.X)                       # scikit-learn feature space
        if self.E is not None:
            sims = 0.5 * sims + 0.5 * cos_to_centroid(self.E)  # + PyTorch learned space
        years = np.array([max(s["year"], BASE_YEAR) for s in self.songs])
        sims = sims + RECENCY_BONUS * np.clip(years - BASE_YEAR, 0, RECENT_SPAN) / RECENT_SPAN   # trend boost

        profile = out["profile"] = self.describe(idx)
        tags = set(profile["genres"] + profile["moods"])
        per, picks = {}, []
        for i in np.argsort(-sims):
            s = self.songs[i]
            if set(s["artists"]) & set(found) or s.get("virtual"):
                continue
            if per.get(s["artist"], 0) >= per_artist:
                continue
            per[s["artist"]] = per.get(s["artist"], 0) + 1
            picks.append({"artist": " & ".join(s["artists"]), "title": s["title"],
                          "year": s["year"], "score": round(float(sims[i]), 2),
                          "style": self.cluster_names[self.labels[i]],
                          "because": [t for t in s["genres"] + s["moods"] if t in tags][:3]})
            if len(picks) == n:
                break
        out["songs"] = picks
        return out

    def explain(self, result):
        """Optional: Claude writes a friendly explanation of the picks."""
        p = result["profile"]
        lines = "\n".join(f"- {s['title']} by {s['artist']} ({', '.join(s['because'])})"
                          for s in result["songs"][:6])
        prompt = (f"A listener likes {', '.join(result['found'])}. Their taste profile: genres "
                  f"{p['genres']}, moods {p['moods']}, traits {p['traits']}. Our model picked:\n"
                  f"{lines}\nIn 3 short sentences, explain what ties these picks to their taste. "
                  "Only use the info given; do not quote lyrics.")
        resp = self.client.messages.create(model=MODEL, max_tokens=250,
                                           messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if b.type == "text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artists", nargs="*", help='e.g. "Bad Bunny, SZA"')
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--explain", action="store_true", help="have Claude explain the picks")
    ap.add_argument("--no-torch", action="store_true", help="scikit-learn only")
    args = ap.parse_args()
    names = [a for a in " ".join(args.artists).split(",") if a.strip()] or \
        [a for a in input("Artists you like (comma-separated): ").split(",") if a.strip()]

    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
    rec = SongRecommender(client=client, use_torch=not args.no_torch)
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
        why = f"  [{', '.join(s['because'])}]" if s["because"] else ""
        print(f"{i:>2}. {s['title']} - {s['artist']} ({s['year']}) {s['score']:.0%}{why}")
    if args.explain and client:
        print("\n" + rec.explain(r))


if __name__ == "__main__":
    main()
