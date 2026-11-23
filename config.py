"""Your preferences live here. Edit and re-run - no other code changes needed."""
from pathlib import Path

DATA_FILE = Path(__file__).parent / "songs.csv"

# Never recommend (or accept as input) these artists / genres.
BLOCKED_ARTISTS = {"taylor swift", "ed sheeran"}
BLOCKED_GENRE_KEYWORDS = {"country", "bluegrass", "honky", "americana"}  # substring match

# Only keep songs from this year on when importing (0 = keep all).
MIN_YEAR = 2020
# Newer songs get a small score boost so the feed leans toward 2025-2026 sounds.
RECENCY_BONUS = 0.05
BASE_YEAR = 2020
RECENT_SPAN = 6  # years over which the recency boost grows (2020 -> 2026)

# Related genres share a "family" token so e.g. alt_pop and bedroom_pop count as similar.
GENRE_FAMILY = {
    "pop": "pop tropical dark_pop dance_pop synth_pop electropop alt_pop indie_pop dream_pop pop_rock "
           "pop_punk kpop soul hyperpop bedroom_pop uk_garage",
    "hiphop": "hip_hop trap pop_rap west_coast rap funk melodic_rap psychedelic_rap trap_soul gospel",
    "rnb": "rnb alt_rnb neo_soul soul afrobeats trap_soul gospel",
    "latin": "reggaeton latin_pop latin_trap salsa",
    "indie": "indie indie_pop psych_pop dream_pop jazz_pop alt_pop",
    "rock": "rock hard_rock pop_rock soft_rock alt_rock indie_rock",
    "dance": "dance_pop house electronic baile_funk psych_pop disco electropop electroclash "
             "hyperpop uk_garage drum_and_bass edm dancehall reggae tropical",
    "asian_pop": "kpop",
}
FAMILIES_OF = {}
for _fam, _gs in GENRE_FAMILY.items():
    for _g in _gs.split():
        FAMILIES_OF.setdefault(_g, []).append("family_" + _fam)

NUMERIC = ["energy", "valence", "dance", "acoustic", "tempo"]
NUM_WEIGHT = 0.8  # audio features vs. genre/mood tags

# How much each "view" of a song counts toward similarity (re-scaled over whatever is available):
#   features = scikit-learn tags + audio features, style = PyTorch autoencoder, text = text embedding
SPACE_WEIGHTS = {"features": 0.4, "style": 0.3, "text": 0.3}
# Likes/dislikes: how strongly the learned preference model moves the ranking.
FEEDBACK_WEIGHT = 0.12
FEEDBACK_MIN_VOTES = 1
