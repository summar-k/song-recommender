"""Everything that talks to Claude (all optional; the app works without a key)."""
import json
import os

MODEL = "claude-sonnet-5-5"


def make_client():
    """Return an Anthropic client if the package and ANTHROPIC_API_KEY are available, else None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # type: ignore  # optional: pip install anthropic
    except ImportError:
        print("Claude features skipped: run `pip install anthropic` to enable them.")
        return None
    return anthropic.Anthropic()


def _ask(client, prompt, max_tokens=600):
    resp = client.messages.create(model=MODEL, max_tokens=max_tokens,
                                  messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in resp.content if b.type == "text")


def _json(text):
    """Pull the first JSON object or array out of a reply."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]")) + 1
    return json.loads(text[start:end])


def infer_artist(client, name):
    """Style description for an artist that isn't in songs.csv (or None if Claude doesn't know them)."""
    d = _json(_ask(client, (
        f'Describe the musical style of the artist "{name}". Reply ONLY with JSON: '
        '{"genres":[3 snake_case genres],"moods":[3 single-word moods],"energy":0-1,'
        '"valence":0-1,"dance":0-1,"acoustic":0-1,"tempo":bpm}. '
        'If you do not know the artist reply {"unknown":true}.'), 300))
    return None if d.get("unknown") else d


def explain_picks(client, found, profile, picks):
    """One short personalised sentence per pick -> {song_id: sentence}."""
    lines = "\n".join(
        f'- id={p["id"]} | "{p["title"]}" by {p["artist"]} ({p["year"]}) | closest to '
        f'{p["closest"]} | shared tags: {", ".join(p["because"]) or "none"} | style: {p["style"]}'
        for p in picks)
    d = _json(_ask(client, (
        f"A listener likes {', '.join(found)}. Taste profile: genres {profile['genres']}, "
        f"moods {profile['moods']}, traits {profile['traits']}.\n"
        f"Our model recommended:\n{lines}\n\n"
        "For each song write ONE sentence (max 22 words) on why they'd like it, naming the "
        "closest artist where natural. Use only the information above. Do not quote lyrics "
        "and do not make claims about the song's content. Reply ONLY with JSON: "
        '{"<id>": "sentence", ...}'), 1500))
    return {str(k): str(v) for k, v in d.items()}


def tag_songs(client, items, genre_vocab, mood_vocab):
    """items: list of (artist, title). Returns a list of dicts (or {"unknown": true}) in order."""
    listing = "\n".join(f"{i + 1}. {a} - {t}" for i, (a, t) in enumerate(items))
    d = _json(_ask(client, (
        "For each song below give its release year and tags for a recommender. Reply ONLY "
        'with a JSON array, one object per song, in order: {"year":int,"genres":[2-4 snake_case],'
        '"moods":[2-3 single words],"energy":0-1,"valence":0-1,"dance":0-1,"acoustic":0-1,'
        '"tempo":bpm,"confident":true/false}. Prefer genres from this list when they fit: '
        f"{', '.join(sorted(genre_vocab))}. Prefer moods from: {', '.join(sorted(mood_vocab))}. "
        'If you do not recognise a song reply {"unknown":true} for it. Do not guess the year '
        'if unsure; set "confident":false instead.\n\n' + listing), 2000))
    return d if isinstance(d, list) else [d]
