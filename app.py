"""The website.   python app.py   then open http://127.0.0.1:5000

Likes/dislikes live in the browser and are sent with each request, so the server
needs no database and no accounts.
"""
from flask import Flask, jsonify, render_template, request

from recommender import SongRecommender

app = Flask(__name__)
rec = SongRecommender()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(text_backend=rec.text_backend, songs=len(rec.songs))


@app.get("/api/artists")
def artists():
    return jsonify(rec.artist_names())


@app.post("/api/recommend")
def recommend():
    body = request.get_json(force=True, silent=True) or {}
    names = [str(a)[:60] for a in body.get("artists", [])][:8]
    return jsonify(rec.recommend(names, n=min(int(body.get("n", 10)), 20),
                                 liked=body.get("liked", [])[:100],
                                 disliked=body.get("disliked", [])[:100]))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
