# afterglow

A song recommender website. Add artists you like and it finds songs that sound like them, then learns from your likes and dislikes.

## Setup (VS Code)

1. Open this folder in VS Code and install the Python and Rainbow CSV extensions.
2. Open the VS Code terminal and create a virtual environment.
3. Activate it and install the requirements (commands below).
4. Pick the interpreter: Cmd/Ctrl+Shift+P, then "Python: Select Interpreter", then `.venv`.

### Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt
and run `.venv\Scripts\activate.bat` instead.

## Run the website

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Run it in the terminal instead

```bash
python recommender.py "Bad Bunny, SZA, Tame Impala"
```

## How it works

- **scikit-learn:** TF-IDF on genre and mood tags, scaled audio features, and KMeans style clusters.
- **PyTorch autoencoder:** learns a 12-number style embedding for each song.
- **Text embeddings:** each song is described in words and embedded with a pretrained model (all-MiniLM-L6-v2). If `sentence-transformers` isn't installed it falls back to TF-IDF + SVD.
- **Learning from feedback:** a small PyTorch logistic regression is trained on your likes and dislikes and re-ranks the list. Votes are stored in your browser, so there is no database.
- **Score:** the three similarity views are blended, plus a small boost for recent songs.

## Import songs

Download a Spotify-style CSV (for example the Kaggle "Spotify tracks dataset"), then run:

```bash
python import_songs.py path/to/file.csv --min-year 2020
```

Open `songs.csv` in VS Code to view or edit it. Blocked artists and genres are filtered out.

## Customize

Edit `config.py` to change blocked artists and genres, the minimum year, the recency boost, the view weights, how strongly feedback counts, and the genre families.

Note: the `year` column is the song's release year, and the audio features are rough estimates.
