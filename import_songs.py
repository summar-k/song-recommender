"""Import songs from a CSV (e.g. a Kaggle 'Spotify tracks' dataset) into songs.csv.

    python import_songs.py path/to/dataset.csv [--min-year 2024] [--max-rows 5000]

Handles common column names automatically, drops blocked artists/genres from config.py
(Taylor Swift, Ed Sheeran, country...), de-duplicates, and auto-creates mood tags.
"""
import argparse

from config import DATA_FILE, MIN_YEAR
from data import import_csv

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--min-year", type=int, default=MIN_YEAR)
ap.add_argument("--max-rows", type=int, default=None)
a = ap.parse_args()
added, skipped = import_csv(a.csv, DATA_FILE, a.min_year, a.max_rows)
print(f"Imported {added} songs, skipped {skipped} (blocked/duplicate/old/invalid).")
