"""The website.   python app.py   then open http://127.0.0.1:5000

Likes/dislikes live in the browser and are sent with each request, so the server
needs no database. Claude features switch on automatically when ANTHROPIC_API_KEY is set.
"""
import os
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, render_template, request

from claude_tools import make_client
from recommender import SongRecommender

app = Flask(__name__)
client = make_client()
rec = SongRecommender(client=client)

# Protect your API bill: at most this many Claude-backed requests per visitor per hour.
CLAUDE_LIMIT_PER_HOUR = int(os.environ.get("CLAUDE_LIMIT_PER_HOUR", "30"))
_hits = defaultdict(deque)


def claude_allowed():
    if not client:
        return False
    q, now = _hits[request.remote_addr], time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= CLAUDE_LIMIT_PER_HOUR:
        return False
    q.append(now)
    return True


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(claude=bool(client), text_backend=rec.text_backend, songs=len(rec.songs))


@app.get("/api/artists")
def artists():
    return jsonify(rec.artist_names())


@app.post("/api/recommend")
def recommend():
    body = request.get_json(force=True, silent=True) or {}
    names = [str(a)[:60] for a in body.get("artists", [])][:8]
    result = rec.recommend(names, n=min(int(body.get("n", 10)), 20),
                           liked=body.get("liked", [])[:100],
                           disliked=body.get("disliked", [])[:100],
                           use_ai=claude_allowed())
    return jsonify(result)


@app.post("/api/explain")
def explain():
    """Claude's personalised one-liners; the page shows the built-in ones first."""
    if not claude_allowed():
        return jsonify(why={})
    body = request.get_json(force=True, silent=True) or {}
    result = rec.recommend([str(a)[:60] for a in body.get("artists", [])][:8], n=10,
                           liked=body.get("liked", [])[:100],
                           disliked=body.get("disliked", [])[:100], use_ai=False)
    if not result["songs"]:
        return jsonify(why={})
    try:
        return jsonify(why=rec.explain(result))
    except Exception:
        return jsonify(why={})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
