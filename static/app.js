(() => {
  const $ = (id) => document.getElementById(id);
  const input = $("input"), chipsEl = $("chips"), suggest = $("suggest"), go = $("go");
  const statusEl = $("status"), resultsEl = $("results"), cardsEl = $("cards");

  let artists = [];            // artists you've added
  let allArtists = [];         // names for suggestions
  let liked = new Set(), disliked = new Set();
  let busy = false, token = 0, active = -1, shown = [];

  const store = {
    get() { try { return JSON.parse(localStorage.getItem("afterglow") || "{}"); } catch (e) { return {}; } },
    set(v) { try { localStorage.setItem("afterglow", JSON.stringify(v)); } catch (e) {} },
  };
  function save() { store.set({ artists, liked: [...liked], disliked: [...disliked] }); }

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  /* ---------- artist chips + suggestions ---------- */
  function renderChips() {
    chipsEl.replaceChildren();
    artists.forEach((a, i) => {
      const chip = el("span", "chip", a);
      const x = el("button", "", "×");
      x.type = "button";
      x.setAttribute("aria-label", "Remove " + a);
      x.onclick = () => { artists.splice(i, 1); save(); renderChips(); };
      chip.appendChild(x);
      chipsEl.appendChild(chip);
    });
    input.placeholder = artists.length ? "add another" : "add an artist you like";
  }

  function addArtist(name) {
    name = name.trim();
    const exact = allArtists.find((a) => norm(a) === norm(name));
    if (exact) name = exact;
    if (!name || artists.some((a) => a.toLowerCase() === name.toLowerCase())) return;
    if (artists.length >= 8) return;
    artists.push(name);
    input.value = "";
    save(); renderChips(); hideSuggest();
  }

  function norm(s) { return s.toLowerCase().replace(/\$/g, "s").replace(/[^a-z0-9 ]/g, ""); }

  function showSuggest() {
    const q = norm(input.value.trim());
    if (!q) return hideSuggest();
    const taken = new Set(artists.map(norm));
    shown = allArtists.filter((a) => norm(a).includes(q) && !taken.has(norm(a))).slice(0, 6);
    suggest.replaceChildren();
    shown.forEach((a, i) => {
      const li = el("li", i === active ? "on" : "", a);
      li.onmousedown = (e) => { e.preventDefault(); addArtist(a); };
      suggest.appendChild(li);
    });
    suggest.hidden = shown.length === 0;
  }
  function hideSuggest() { suggest.hidden = true; active = -1; }

  input.addEventListener("input", () => { active = -1; showSuggest(); });
  input.addEventListener("blur", hideSuggest);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" && shown.length) { e.preventDefault(); active = (active + 1) % shown.length; showSuggest(); }
    else if (e.key === "ArrowUp" && shown.length) { e.preventDefault(); active = (active - 1 + shown.length) % shown.length; showSuggest(); }
    else if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      if (active >= 0 && shown[active]) addArtist(shown[active]);
      else if (input.value.trim()) addArtist(input.value);
      else if (artists.length) run();
    } else if (e.key === "Backspace" && !input.value && artists.length) {
      artists.pop(); save(); renderChips();
    }
  });
  go.onclick = () => { if (input.value.trim()) addArtist(input.value); run(); };
  $("field").onclick = () => input.focus();

  /* ---------- talking to the server ---------- */
  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error("server error");
    return r.json();
  }

  function setStatus(msg, error) {
    statusEl.hidden = !msg;
    statusEl.textContent = msg || "";
    statusEl.className = "status" + (error ? " error" : "");
  }

  async function run() {
    if (!artists.length) { setStatus("add at least one artist first.", true); return; }
    const mine = ++token;
    busy = true; go.disabled = true;
    if (resultsEl.hidden) setStatus("finding your sound…");
    try {
      const body = { artists, liked: [...liked], disliked: [...disliked], n: 10 };
      const res = await post("/api/recommend", body);
      if (mine !== token) return;
      render(res);
      explainWithClaude(body, mine);
    } catch (e) {
      setStatus("something went wrong. is the server still running?", true);
    } finally {
      if (mine === token) { busy = false; go.disabled = false; }
    }
  }

  async function explainWithClaude(body, mine) {
    if (!window.CLAUDE_ON) return;
    try {
      const { why } = await post("/api/explain", body);
      if (mine !== token) return;
      document.querySelectorAll(".card").forEach((c) => {
        if (why[c.dataset.id]) c.querySelector(".why").textContent = why[c.dataset.id];
      });
    } catch (e) { /* the built-in sentence stays */ }
  }

  /* ---------- drawing results ---------- */
  function row(label, items) {
    const r = el("div", "row");
    r.appendChild(el("span", "label", label));
    items.forEach((t) => r.appendChild(el("span", "tag", t.replace(/_/g, " "))));
    return r;
  }

  function render(res) {
    const notes = [];
    if (res.blocked.length) notes.push("skipped: " + res.blocked.join(", ") + " (blocked in config.py)");
    if (res.missing.length) notes.push("not recognized: " + res.missing.join(", "));
    setStatus(notes.join("  ·  "), false);
    if (!res.profile) { resultsEl.hidden = true; return; }

    const p = res.profile;
    const taste = $("taste");
    taste.replaceChildren(
      row("genres", p.genres), row("moods", p.moods),
      row("vibe", p.traits.concat(["~" + p.avg_tempo + " bpm"])));

    cardsEl.replaceChildren();
    res.songs.forEach((s, i) => {
      const li = el("li", "card" + (liked.has(s.id) ? " liked" : ""));
      li.dataset.id = s.id;
      li.style.animationDelay = i * 40 + "ms";
      li.appendChild(el("div", "rank", String(i + 1)));

      const body = el("div");
      body.appendChild(el("h3", "title", s.title));
      body.appendChild(el("p", "meta", s.artist + "  ·  " + s.year));
      body.appendChild(el("p", "why", s.why));
      const tags = el("div", "tags");
      s.because.forEach((t) => tags.appendChild(el("span", "tag", t.replace(/_/g, " "))));
      if (tags.children.length) body.appendChild(tags);
      li.appendChild(body);

      const m = el("div", "match", Math.min(99, Math.round(s.score * 100)) + "%");
      m.appendChild(el("small", "", "match"));
      li.appendChild(m);

      const actions = el("div", "actions");
      const like = el("button", "vote like" + (liked.has(s.id) ? " on" : ""), liked.has(s.id) ? "Liked" : "Like");
      const dislike = el("button", "vote dislike", "Dislike");
      like.type = dislike.type = "button";
      like.onclick = () => vote(s.id, "like");
      dislike.onclick = () => vote(s.id, "dislike");
      actions.append(like, dislike);
      li.appendChild(actions);
      cardsEl.appendChild(li);
    });
    $("reset").hidden = !(liked.size || disliked.size);
    resultsEl.hidden = false;
  }

  function vote(id, kind) {
    if (kind === "like") { liked.has(id) ? liked.delete(id) : liked.add(id); disliked.delete(id); }
    else { disliked.add(id); liked.delete(id); }
    save(); run();
  }
  $("reset").onclick = () => { liked.clear(); disliked.clear(); save(); run(); };

  /* ---------- start up ---------- */
  async function init() {
    const saved = store.get();
    artists = saved.artists || []; liked = new Set(saved.liked || []); disliked = new Set(saved.disliked || []);
    renderChips();
    try {
      const [names, st] = await Promise.all([fetch("/api/artists").then((r) => r.json()), fetch("/api/status").then((r) => r.json())]);
      allArtists = names;
      window.CLAUDE_ON = st.claude;
      const foot = $("footer");
      foot.replaceChildren(
        el("div", "", st.songs + " songs  ·  scikit-learn + PyTorch + text embeddings"),
        el("div", "", st.claude ? "Claude explanations on" : "Claude explanations off: set ANTHROPIC_API_KEY to turn them on"));
    } catch (e) { /* offline: typing still works */ }
    if (artists.length) run();
  }
  init();
})();
