"""Claude fills in release year, genres, moods and audio features for songs you add.

    python autotag.py "Kaytranada - Witchy" "SZA - Saturn"
    python autotag.py --file new_songs.txt        # one "Artist - Title" per line
    python autotag.py "Artist - Title" --yes      # skip the confirmation prompt

Claude's numbers are estimates, and its release years can be wrong, so check the
table it prints before saying yes. Needs ANTHROPIC_API_KEY.
"""
import argparse
import csv
import re
import sys

import data
from claude_tools import make_client, tag_songs
from config import DATA_FILE, MIN_YEAR, NUMERIC


def parse_items(lines):
    items = []
    for line in lines:
        if " - " in line:
            a, t = line.split(" - ", 1)
            items.append((a.strip(), t.strip()))
        elif line.strip():
            print(f'Skipping "{line}": use the form "Artist - Title"')
    return items


def clean_tags(tags):
    return [re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s\-/]+", "_", str(t).strip().lower()))
            for t in tags if str(t).strip()]


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("songs", nargs="*", help='"Artist - Title"')
    ap.add_argument("--file", help="text file with one 'Artist - Title' per line")
    ap.add_argument("--yes", action="store_true", help="don't ask before saving")
    args = ap.parse_args()
    lines = list(args.songs)
    if args.file:
        lines += open(args.file, encoding="utf-8").read().splitlines()
    items = parse_items(lines)
    if not items:
        sys.exit('Give songs like: python autotag.py "Kaytranada - Witchy"')

    client = make_client()
    if not client:
        sys.exit("Auto-tagging needs the anthropic package and ANTHROPIC_API_KEY set.")

    existing = data.load_songs(DATA_FILE)
    have = {(s["artist"].casefold(), s["title"].casefold()) for s in existing}
    genre_vocab = {g for s in existing for g in s["genres"]}
    mood_vocab = {m for s in existing for m in s["moods"]}

    results = tag_songs(client, items, genre_vocab, mood_vocab)
    rows = []
    for (artist, title), r in zip(items, results):
        if (artist.casefold(), title.casefold()) in have:
            print(f"Already in songs.csv: {artist} - {title}")
        elif r.get("unknown"):
            print(f"Claude doesn't recognise: {artist} - {title}")
        else:
            try:
                genres, moods = clean_tags(r["genres"]), clean_tags(r["moods"])
                nums = [clamp(r[k], 0, 1) for k in NUMERIC[:4]] + [clamp(r["tempo"], 40, 220)]
                year = int(r["year"])
            except (KeyError, ValueError, TypeError):
                print(f"Couldn't read Claude's answer for: {artist} - {title}")
                continue
            if not data.is_allowed([artist], genres):
                print(f"Blocked by config.py: {artist} - {title}")
                continue
            if MIN_YEAR and year < MIN_YEAR:
                print(f"Skipping {artist} - {title}: released {year}, before {MIN_YEAR}")
                continue
            rows.append(dict(artist=artist, title=title, year=year, genres=" ".join(genres),
                             moods=" ".join(moods), **dict(zip(NUMERIC, nums)),
                             _sure=r.get("confident", True)))
    if not rows:
        return
    print("\n" + f"{'song':<44}{'year':<6}{'genres':<32}moods / energy valence dance acoustic bpm")
    for r in rows:
        flag = "" if r["_sure"] else "   <- Claude is NOT sure: check the year"
        print(f"{r['artist'] + ' - ' + r['title']:<44}{r['year']:<6}{r['genres']:<32}"
              f"{r['moods']} / {r['energy']:.2f} {r['valence']:.2f} {r['dance']:.2f} "
              f"{r['acoustic']:.2f} {r['tempo']:.0f}{flag}")
    if not args.yes and input("\nAdd these to songs.csv? [y/N] ").strip().lower() != "y":
        return

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
        fields = list(all_rows[0].keys())
    for r in rows:
        r.pop("_sure")
        all_rows.append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    prim = lambda r: r["artist"].split(";")[0].strip().casefold()
    all_rows.sort(key=lambda r: (prim(r), int(r["year"] or 0), r["title"].casefold()))
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Added {len(rows)} song(s) to songs.csv (sorted artist, year, title).")


if __name__ == "__main__":
    main()
