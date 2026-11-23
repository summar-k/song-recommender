# Trend-Aware Song Recommender

Built with scikit-learn and PyTorch.

## Setup (VS Code)

1. Open this folder in VS Code and install the Python and Rainbow CSV extensions.
2. Open the VS Code terminal and create a virtual environment.
3. Activate it and install the requirements (commands below).
4. Pick the interpreter: Cmd/Ctrl+Shift+P, then "Python: Select Interpreter", then `.venv`.
5. Run the recommender (or press F5 to debug).

### Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python recommender.py "Bad Bunny, SZA, Tame Impala"
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python recommender.py "Bad Bunny, SZA, Tame Impala"
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt
and run `.venv\Scripts\activate.bat` instead.

## Import songs

Download a Spotify-style CSV (for example the Kaggle "Spotify tracks dataset"), then run:

```bash
python import_songs.py path/to/file.csv --min-year 2020
```

Open `songs.csv` in VS Code to view or edit it. Blocked artists and genres are filtered out.

## Customize

Edit `config.py` to change blocked artists and genres, the minimum year, the recency boost and the genre families.

## How it works

- **scikit-learn:** TF-IDF on genre and mood tags, scaled audio features, and KMeans style clusters.
- **PyTorch:** a denoising autoencoder learns a 12-number style embedding for each song.
- **Claude (optional):** set `ANTHROPIC_API_KEY` to let Claude infer unknown artists and use `--explain` for a short summary of the picks. Without a key, everything else still works.
- **Score:** 50% feature similarity, 50% embedding similarity, plus a small boost for recent songs.
Note: the `year` column is the song's release year, and the audio features are rough estimates.
