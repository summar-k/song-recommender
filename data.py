"""Loading, filtering and importing song data (works with Kaggle-style Spotify CSVs)."""
import csv
import re
from pathlib import Path

from config import BLOCKED_ARTISTS, BLOCKED_GENRE_KEYWORDS, MIN_YEAR, NUMERIC

FIELDS = ["artist", "title", "year", "genres", "moods"] + NUMERIC

# Column-name aliases used by common datasets (Kaggle Spotify tracks, etc.)
ALIASES = {
    "artist": ["artist", "artists", "artist_name", "track_artist"],
    "title": ["title", "track_name", "name", "song", "track"],
    "genres": ["genres", "track_genre", "genre", "playlist_genre", "artist_genres"],
    "moods": ["moods"],
    "energy": ["energy"],
    "valence": ["valence"],
    "dance": ["dance", "danceability"],
    "acoustic": ["acoustic", "acousticness"],
    "tempo": ["tempo", "bpm"],
    "year": ["year", "release_year", "release_date", "album_release_date"],
}


def norm_genres(text):
    parts = re.split(r"[;,]", text or "") if re.search(r"[;,]", text or "") else [text or ""]
    out = []
    for p in parts:
        p = re.sub(r"[\s\-&/]+", "_", p.strip().lower()).strip("_")
        out += [t for t in p.split() if t] if " " in p else ([p] if p else [])
    return out


def is_allowed(artists, genres):
    if any(a.casefold() in BLOCKED_ARTISTS for a in artists):
        return False
    return not any(k in g for g in genres for k in BLOCKED_GENRE_KEYWORDS)


def auto_moods(energy, valence, dance, acoustic):
    """Rule-based mood tags for imported rows that have none."""
    m = []
    m.append("energetic" if energy > .7 else "chill" if energy < .45 else "balanced")
    m.append("upbeat" if valence > .65 else "melancholic" if valence < .35 else "moody")
    if dance > .75:
        m.append("groovy")
    if acoustic > .6:
        m.append("tender")
    return m


def _make(artist, title, year, genres, moods, nums):
    artists = [a.strip() for a in re.split(r"[;]", artist) if a.strip()]
    return dict(artists=artists, artist=artists[0], title=title, year=int(year or 0),
                genres=genres, moods=moods, **dict(zip(NUMERIC, nums)))


def load_songs(path):
    songs = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = _make(r["artist"], r["title"], r["year"], r["genres"].split(),
                      r["moods"].split(), [float(r[k]) for k in NUMERIC])
            if is_allowed(s["artists"], s["genres"]):
                songs.append(s)
    return songs


def _pick(header, key):
    lower = {h.lower(): h for h in header}
    return next((lower[a] for a in ALIASES[key] if a in lower), None)


def import_csv(src, dest, min_year=MIN_YEAR, max_rows=None):
    """Merge any Spotify-style CSV into the project's songs.csv. Returns (added, skipped)."""
    existing = {(s["artist"].casefold(), s["title"].casefold()) for s in load_songs(dest)} \
        if Path(dest).exists() else set()
    added, skipped, rows = 0, 0, []
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = {k: _pick(reader.fieldnames, k) for k in ALIASES}
        missing = [k for k in ("artist", "title", "energy", "valence") if not col[k]]
        if missing:
            raise ValueError(f"CSV is missing columns for: {missing}. Found: {reader.fieldnames}")
        for r in reader:
            try:
                artist = r[col["artist"]].replace(",", ";")
                title = r[col["title"]]
                genres = norm_genres(r[col["genres"]]) if col["genres"] else []
                year = int(re.search(r"(19|20)\d\d", r[col["year"]] or "").group()) \
                    if col["year"] and re.search(r"(19|20)\d\d", r[col["year"]] or "") else 0
                e, v = float(r[col["energy"]]), float(r[col["valence"]])
                d = float(r[col["dance"]]) if col["dance"] else .5
                a = float(r[col["acoustic"]]) if col["acoustic"] else .3
                t = float(r[col["tempo"]]) if col["tempo"] else 110
            except (ValueError, TypeError, AttributeError):
                skipped += 1
                continue
            s = _make(artist, title, year, genres or ["pop"], [], [e, v, d, a, t])
            key = (s["artist"].casefold(), title.casefold())
            if (not is_allowed(s["artists"], s["genres"]) or key in existing
                    or (min_year and year and year < min_year)):
                skipped += 1
                continue
            s["moods"] = auto_moods(e, v, d, a)
            existing.add(key)
            rows.append(s)
            added += 1
            if max_rows and added >= max_rows:
                break
    new_file = not Path(dest).exists()
    with open(dest, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        for s in rows:
            w.writerow([";".join(s["artists"]), s["title"], s["year"], " ".join(s["genres"]),
                        " ".join(s["moods"])] + [s[k] for k in NUMERIC])
    return added, skipped
