"use strict";

const COLORS = {
  bg: "#0a0c10",
  rowDark: "rgba(255,255,255,0.025)",
  grid: "rgba(255,255,255,0.06)",
  gridMajor: "rgba(255,255,255,0.12)",
  label: "rgba(255,255,255,0.45)",
  game: "rgba(255,160,70,0.80)",
  gameActive: "rgba(255,190,120,1)",
  gameText: "#1a0e02",
  ref: "rgba(140,120,255,0.95)",
  refFill: "rgba(140,120,255,0.12)",
  wave: "rgba(0,218,232,0.45)",
  word: "rgba(255,255,255,0.07)",
  wordActive: "rgba(0,218,232,0.22)",
  phon: "rgba(0,218,232,0.16)",
  phonActive: "rgba(0,218,232,0.55)",
  playhead: "rgba(0,218,232,0.95)",
};

const KEYS_W = 46;
const RULER_H = 22;
const WAVE_H = 54;
const WORD_H = 24;
const PHON_H = 30;
const LANE_GAP = 4;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BLACK = new Set([1, 3, 6, 8, 10]);

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const canvas = $("canvas");
const ctx2d = canvas.getContext("2d");

const state = {
  examples: [],
  result: null,
  reference: [],
  notes: [],
  phonemes: [],
  words: [],
  peaks: null,
  duration: 0,
  t0: 0,
  pxPerSec: 90,
  pitchLo: 48,
  pitchHi: 72,
  activePhoneme: -1,
};

const synth = { ctx: null, voiceGain: null, synthGain: null, voices: [] };

function noteName(midi) {
  return NOTE_NAMES[midi % 12] + (Math.floor(midi / 12) - 1);
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function wordSpans(phonemes) {
  const spans = [];
  for (const p of phonemes) {
    const last = spans[spans.length - 1];
    if (last && last.word === p.word && p.start - last.end < 0.6) {
      last.end = Math.max(last.end, p.end);
    } else {
      spans.push({ word: p.word || "", start: p.start, end: p.end });
    }
  }
  return spans;
}

function noteAt(notes, t) {
  return notes.findIndex((n) => t >= n.onset && t < n.offset);
}

function phonemeAt(t) {
  let best = -1;
  for (let i = 0; i < state.phonemes.length; i++) {
    if (state.phonemes[i].start <= t) best = i;
    else break;
  }
  return best;
}

function frameMetrics(ref, est, duration) {
  const step = 0.01;
  let voiced = 0;
  let hit = 0;
  let exact = 0;
  for (let t = 0; t < duration; t += step) {
    const r = ref.find((n) => t >= n.onset && t < n.offset);
    if (!r) continue;
    voiced++;
    const e = est.find((n) => t >= n.onset && t < n.offset);
    if (!e) continue;
    hit++;
    if (e.pitch === r.pitch) exact++;
  }
  return voiced ? { recall: hit / voiced, accuracy: exact / voiced } : null;
}

// ---------- layout ----------

function layout() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const lanesH = WAVE_H + WORD_H + PHON_H + LANE_GAP * 3;
  const roll = { x: KEYS_W, y: RULER_H, w: w - KEYS_W, h: h - RULER_H - lanesH };
  const wave = { y: roll.y + roll.h + LANE_GAP, h: WAVE_H };
  const word = { y: wave.y + WAVE_H + LANE_GAP, h: WORD_H };
  const phon = { y: word.y + WORD_H + LANE_GAP, h: PHON_H };
  return { w, h, roll, wave, word, phon };
}

function visibleDur(L) {
  return L.roll.w / state.pxPerSec;
}

function tx(t) {
  return KEYS_W + (t - state.t0) * state.pxPerSec;
}

function xt(x) {
  return state.t0 + (x - KEYS_W) / state.pxPerSec;
}

function clampView(L) {
  const maxT0 = Math.max(0, state.duration - visibleDur(L));
  state.t0 = Math.min(Math.max(0, state.t0), maxT0);
  const scroll = $("scroll");
  scroll.value = maxT0 > 0 ? Math.round((state.t0 / maxT0) * 1000) : 0;
  scroll.disabled = maxT0 === 0;
}

// ---------- drawing ----------

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function rulerStep() {
  const candidates = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60];
  return candidates.find((s) => s * state.pxPerSec >= 60) || 60;
}

function draw() {
  const L = layout();
  clampView(L);
  const c = ctx2d;
  c.fillStyle = COLORS.bg;
  c.fillRect(0, 0, L.w, L.h);
  if (!state.result) return;

  const t1 = state.t0 + visibleDur(L);
  const rows = state.pitchHi - state.pitchLo + 1;
  const rowH = L.roll.h / rows;
  const py = (p) => L.roll.y + (state.pitchHi - p) * rowH;

  c.save();
  c.beginPath();
  c.rect(0, 0, L.w, L.h);
  c.clip();

  // piano rows
  for (let p = state.pitchLo; p <= state.pitchHi; p++) {
    const y = py(p);
    if (BLACK.has(p % 12)) {
      c.fillStyle = COLORS.rowDark;
      c.fillRect(KEYS_W, y, L.roll.w, rowH);
    }
    if (p % 12 === 0) {
      c.fillStyle = COLORS.gridMajor;
      c.fillRect(KEYS_W, y + rowH - 1, L.roll.w, 1);
    }
  }

  // time grid + ruler
  const step = rulerStep();
  c.font = "11px Inter, system-ui, sans-serif";
  c.textBaseline = "middle";
  for (let t = Math.floor(state.t0 / step) * step; t <= t1; t += step) {
    const x = Math.round(tx(t)) + 0.5;
    if (x < KEYS_W) continue;
    c.fillStyle = COLORS.grid;
    c.fillRect(x, RULER_H, 1, L.h - RULER_H);
    c.fillStyle = COLORS.label;
    c.fillText(step < 1 ? t.toFixed(step < 0.5 ? 2 : 1) + "s" : fmtTime(t).replace(/\.0$/, ""), x + 4, RULER_H / 2);
  }

  // reference notes
  if ($("showRef").checked) {
    for (const n of state.reference) {
      if (n.offset < state.t0 || n.onset > t1) continue;
      const x = tx(n.onset);
      const w = Math.max(2, (n.offset - n.onset) * state.pxPerSec);
      const y = py(n.pitch);
      c.fillStyle = COLORS.refFill;
      c.fillRect(x, y, w, rowH);
      c.strokeStyle = COLORS.ref;
      c.lineWidth = 1.5;
      c.strokeRect(x + 0.75, y + 0.75, w - 1.5, rowH - 1.5);
    }
  }

  // GAME notes
  const now = audio.currentTime;
  if ($("showGame").checked) {
    c.textBaseline = "middle";
    for (const n of state.notes) {
      if (n.offset < state.t0 || n.onset > t1) continue;
      const x = tx(n.onset);
      const w = Math.max(2, (n.offset - n.onset) * state.pxPerSec - 1);
      const y = py(n.pitch) + 1;
      const h = Math.max(2, rowH - 2);
      const active = now >= n.onset && now < n.offset;
      c.fillStyle = active ? COLORS.gameActive : COLORS.game;
      roundRect(c, x, y, w, h, Math.min(3, h / 2));
      c.fill();
      const label = n.phonemes.join(" ");
      if (label && w > 18 && h >= 11) {
        c.fillStyle = COLORS.gameText;
        c.font = `${Math.min(12, h - 2)}px Inter, system-ui, sans-serif`;
        c.save();
        c.beginPath();
        c.rect(x, y, w, h);
        c.clip();
        c.fillText(label, x + 4, y + h / 2 + 0.5);
        c.restore();
      }
    }
  }

  // key labels
  c.fillStyle = "#0e1117";
  c.fillRect(0, RULER_H, KEYS_W, L.roll.h);
  c.font = "10px Inter, system-ui, sans-serif";
  for (let p = state.pitchLo; p <= state.pitchHi; p++) {
    const y = py(p);
    c.fillStyle = BLACK.has(p % 12) ? "#1a1e27" : "#262b36";
    c.fillRect(0, y + 0.5, KEYS_W - 6, rowH - 1);
    if (p % 12 === 0 || rowH >= 13) {
      c.fillStyle = p % 12 === 0 ? "rgba(255,255,255,0.8)" : COLORS.label;
      c.fillText(noteName(p), 4, y + rowH / 2);
    }
  }

  drawWave(c, L, t1);
  drawLanes(c, L, t1, now);

  // playhead
  const x = tx(now);
  if (x >= KEYS_W && x <= L.w) {
    c.fillStyle = COLORS.playhead;
    c.fillRect(Math.round(x) - 1, 0, 2, L.h);
  }
  c.restore();
}

function drawWave(c, L, t1) {
  c.fillStyle = "#0e1117";
  c.fillRect(0, L.wave.y, KEYS_W, L.wave.h);
  laneLabel(c, "onda", L.wave);
  if (!state.peaks) return;
  const { min, max, rate } = state.peaks;
  const mid = L.wave.y + L.wave.h / 2;
  const amp = L.wave.h / 2 - 2;
  c.fillStyle = COLORS.wave;
  for (let x = KEYS_W; x < L.w; x++) {
    const ta = xt(x);
    const tb = xt(x + 1);
    if (tb < 0 || ta > state.duration) continue;
    const a = Math.max(0, Math.floor(ta * rate));
    const b = Math.min(min.length, Math.max(a + 1, Math.ceil(tb * rate)));
    let lo = 0;
    let hi = 0;
    for (let i = a; i < b; i++) {
      if (min[i] < lo) lo = min[i];
      if (max[i] > hi) hi = max[i];
    }
    c.fillRect(x, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
  }
}

function drawLanes(c, L, t1, now) {
  c.textBaseline = "middle";
  c.fillStyle = "#0e1117";
  c.fillRect(0, L.word.y, KEYS_W, L.word.h);
  c.fillRect(0, L.phon.y, KEYS_W, L.phon.h);
  laneLabel(c, "palavra", L.word);
  laneLabel(c, "fonema", L.phon);

  c.font = "12px Inter, system-ui, sans-serif";
  for (const s of state.words) {
    if (s.end < state.t0 || s.start > t1) continue;
    const x = Math.max(KEYS_W, tx(s.start));
    const w = Math.max(2, tx(s.end) - x);
    const active = now >= s.start && now < s.end;
    c.fillStyle = active ? COLORS.wordActive : COLORS.word;
    roundRect(c, x, L.word.y, w, L.word.h, 4);
    c.fill();
    c.fillStyle = "rgba(255,255,255,0.85)";
    clipText(c, s.word, x, L.word.y, w, L.word.h);
  }

  c.font = "13px Inter, system-ui, sans-serif";
  state.phonemes.forEach((p, i) => {
    if (p.end < state.t0 || p.start > t1) return;
    const x = tx(p.start);
    const next = state.phonemes[i + 1];
    // MMS phonemes are point-like (~20 ms); widen the label box up to the next one.
    const labelEnd = Math.max(p.end, Math.min(next ? next.start : p.end + 0.25, p.start + 0.25));
    const w = Math.max(2, tx(labelEnd) - x - 1);
    const active = i === state.activePhoneme;
    c.fillStyle = active ? COLORS.phonActive : COLORS.phon;
    roundRect(c, x, L.phon.y, w, L.phon.h, 4);
    c.fill();
    c.fillStyle = active ? "#031417" : "rgba(255,255,255,0.9)";
    clipText(c, p.phoneme, x, L.phon.y, w, L.phon.h);
  });
}

function laneLabel(c, text, lane) {
  c.fillStyle = COLORS.label;
  c.font = "10px Inter, system-ui, sans-serif";
  c.textBaseline = "middle";
  c.fillText(text, 4, lane.y + lane.h / 2);
}

function clipText(c, text, x, y, w, h) {
  if (w < 10 || !text) return;
  c.save();
  c.beginPath();
  c.rect(x, y, w, h);
  c.clip();
  c.fillText(text, x + 4, y + h / 2 + 0.5);
  c.restore();
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

// ---------- audio ----------

function ensureAudioGraph() {
  if (synth.ctx) return;
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  synth.ctx = ac;
  synth.voiceGain = ac.createGain();
  synth.synthGain = ac.createGain();
  synth.voiceGain.gain.value = Number($("volVoice").value);
  synth.synthGain.gain.value = Number($("volSynth").value);
  ac.createMediaElementSource(audio).connect(synth.voiceGain).connect(ac.destination);
  synth.synthGain.connect(ac.destination);
}

function stopSynth() {
  for (const v of synth.voices) {
    try { v.stop(); } catch (_) { /* already stopped */ }
  }
  synth.voices = [];
}

function scheduleSynth() {
  stopSynth();
  if (!synth.ctx || audio.paused) return;
  const ac = synth.ctx;
  const base = ac.currentTime + 0.03;
  const pos = audio.currentTime;
  for (const n of state.notes) {
    if (n.offset <= pos) continue;
    const start = base + Math.max(0, n.onset - pos);
    const end = base + (n.offset - pos);
    const osc = ac.createOscillator();
    const env = ac.createGain();
    osc.type = "triangle";
    osc.frequency.value = 440 * Math.pow(2, (n.pitch - 69) / 12);
    env.gain.setValueAtTime(0, start);
    env.gain.linearRampToValueAtTime(0.5, start + 0.015);
    env.gain.setValueAtTime(0.5, Math.max(start + 0.015, end - 0.03));
    env.gain.linearRampToValueAtTime(0, end);
    osc.connect(env).connect(synth.synthGain);
    osc.start(start);
    osc.stop(end + 0.01);
    synth.voices.push(osc);
  }
}

async function togglePlay() {
  if (!state.result) return;
  ensureAudioGraph();
  if (synth.ctx.state === "suspended") await synth.ctx.resume();
  if (audio.paused) await audio.play();
  else audio.pause();
}

function seek(t) {
  audio.currentTime = Math.min(Math.max(0, t), state.duration);
  scheduleSynth();
  tick(true);
}

async function computePeaks(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const ac = new OfflineAudioContext(1, 1, 44100);
  const decoded = await ac.decodeAudioData(buf);
  const data = decoded.getChannelData(0);
  const rate = 200;
  const per = Math.max(1, Math.floor(decoded.sampleRate / rate));
  const n = Math.ceil(data.length / per);
  const min = new Float32Array(n);
  const max = new Float32Array(n);
  let peak = 1e-6;
  for (let i = 0; i < n; i++) {
    let lo = 0;
    let hi = 0;
    for (let j = i * per, e = Math.min(data.length, j + per); j < e; j++) {
      const v = data[j];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    min[i] = lo;
    max[i] = hi;
    peak = Math.max(peak, hi, -lo);
  }
  for (let i = 0; i < n; i++) {
    min[i] /= peak;
    max[i] /= peak;
  }
  return { min, max, rate: decoded.sampleRate / per, duration: decoded.duration };
}

// ---------- UI sync ----------

function tick(force = false) {
  const t = audio.currentTime;
  $("time").textContent = `${fmtTime(t)} / ${fmtTime(state.duration)}`;
  const L = layout();
  if ($("follow").checked && !audio.paused) {
    const vis = visibleDur(L);
    if (t > state.t0 + vis * 0.85 || t < state.t0) state.t0 = t - vis * 0.15;
  }
  const idx = phonemeAt(t);
  if (idx !== state.activePhoneme || force) {
    state.activePhoneme = idx;
    renderNow(t);
    highlightRow(idx);
  }
  draw();
}

function loop() {
  tick();
  if (!audio.paused) requestAnimationFrame(loop);
}

function renderNow(t) {
  const p = state.phonemes[state.activePhoneme];
  const ni = noteAt(state.notes, t);
  const n = state.notes[ni];
  const ri = noteAt(state.reference, t);
  const r = state.reference[ri];
  $("now").innerHTML = `
    <div><span class="k">Fonema</span><span class="big ph">${p ? escapeHtml(p.phoneme) : "—"}</span></div>
    <div><span class="k">Palavra</span><span class="big">${p ? escapeHtml(p.word || "") : "—"}</span></div>
    <div><span class="k">Nota GAME</span><span class="big nt">${n ? noteName(n.pitch) : "—"}</span></div>
    ${state.reference.length ? `<div><span class="k">Referência</span><span class="big" style="color:var(--ref)">${r ? noteName(r.pitch) : "—"}</span></div>` : ""}`;
}

function highlightRow(idx) {
  const body = $("phonemes").tBodies[0];
  const prev = body.querySelector("tr.active");
  if (prev) prev.classList.remove("active");
  const row = body.rows[idx];
  if (!row) return;
  row.classList.add("active");
  if (!audio.paused) {
    const wrap = row.closest(".table-wrap");
    const top = row.offsetTop - wrap.clientHeight / 2;
    wrap.scrollTop = Math.max(0, top);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);
}

function renderSummary(ex) {
  const r = state.result;
  const cards = [
    ["Idioma", r.language],
    ["Alinhador", r.aligner.toUpperCase()],
    ["Fonemas", r.phonemes.length],
    ["Palavras", state.words.length],
    ["Notas GAME", state.notes.length],
    ["GAME lang id", r.game.language_id],
    ["Duração", fmtTime(state.duration)],
  ];
  if (state.reference.length) {
    const m = frameMetrics(state.reference, state.notes, state.duration);
    if (m) {
      cards.push(["Pitch exato vs ref.", `${(m.accuracy * 100).toFixed(1)}%`]);
      cards.push(["Voz detectada", `${(m.recall * 100).toFixed(1)}%`]);
    }
  }
  $("summary").innerHTML = cards
    .map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${escapeHtml(v)}</div></div>`)
    .join("");
  $("credit").textContent = `${ex.title} — ${ex.credit}`;
}

function renderTable() {
  const body = $("phonemes").tBodies[0];
  body.innerHTML = state.phonemes
    .map((p, i) => {
      const n = state.notes[noteAt(state.notes, (p.start + p.end) / 2)];
      return `<tr data-i="${i}"><td>${i + 1}</td><td>${p.start.toFixed(3)}</td><td>${p.end.toFixed(3)}</td>` +
        `<td><b>${escapeHtml(p.phoneme)}</b></td><td>${escapeHtml(p.word || "")}</td><td>${n ? noteName(n.pitch) : "—"}</td></tr>`;
    })
    .join("");
}

async function loadExample(ex) {
  $("loading").hidden = false;
  $("play").disabled = true;
  audio.pause();
  stopSynth();
  const [result, reference] = await Promise.all([
    fetch(ex.result).then((r) => r.json()),
    ex.reference ? fetch(ex.reference).then((r) => r.json()) : Promise.resolve([]),
  ]);
  state.result = result;
  state.reference = reference;
  state.notes = result.game.notes;
  state.phonemes = [...result.phonemes].sort((a, b) => a.start - b.start);
  state.words = wordSpans(state.phonemes);
  const pitches = [...state.notes, ...reference].map((n) => n.pitch);
  const lo = Math.min(...pitches);
  const hi = Math.max(...pitches);
  const pad = Math.max(2, Math.ceil((14 - (hi - lo)) / 2));
  state.pitchLo = lo - pad;
  state.pitchHi = hi + pad;
  audio.src = ex.audio;
  state.peaks = await computePeaks(ex.audio);
  state.duration = state.peaks.duration;
  state.t0 = 0;
  state.activePhoneme = -1;
  renderSummary(ex);
  renderTable();
  $("loading").hidden = true;
  $("play").disabled = false;
  tick(true);
}

// ---------- events ----------

function bind() {
  $("play").addEventListener("click", togglePlay);
  audio.addEventListener("play", () => {
    $("play").textContent = "❚❚ Pause";
    scheduleSynth();
    requestAnimationFrame(loop);
  });
  audio.addEventListener("pause", () => {
    $("play").textContent = "▶ Play";
    stopSynth();
    tick(true);
  });
  audio.addEventListener("seeked", () => tick(true));

  $("volVoice").addEventListener("input", (e) => synth.voiceGain && (synth.voiceGain.gain.value = Number(e.target.value)));
  $("volSynth").addEventListener("input", (e) => synth.synthGain && (synth.synthGain.gain.value = Number(e.target.value)));
  $("zoom").addEventListener("input", (e) => {
    const L = layout();
    const center = state.t0 + visibleDur(L) / 2;
    state.pxPerSec = Number(e.target.value);
    state.t0 = center - visibleDur(L) / 2;
    draw();
  });
  $("scroll").addEventListener("input", (e) => {
    const L = layout();
    state.t0 = (Number(e.target.value) / 1000) * Math.max(0, state.duration - visibleDur(L));
    draw();
  });
  for (const id of ["showGame", "showRef"]) $(id).addEventListener("change", draw);

  canvas.addEventListener("click", (e) => {
    const x = e.offsetX;
    if (x < KEYS_W) return;
    seek(xt(x));
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      const anchor = xt(e.offsetX);
      const z = Math.min(400, Math.max(20, state.pxPerSec * Math.exp(-e.deltaY * 0.002)));
      state.pxPerSec = z;
      $("zoom").value = String(Math.round(z));
      state.t0 = anchor - (e.offsetX - KEYS_W) / z;
    } else {
      const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      state.t0 += d / state.pxPerSec;
    }
    draw();
  }, { passive: false });
  canvas.addEventListener("mousemove", hover);
  canvas.addEventListener("mouseleave", () => ($("tooltip").hidden = true));

  $("phonemes").tBodies[0].addEventListener("click", (e) => {
    const row = e.target.closest("tr");
    if (!row) return;
    const p = state.phonemes[Number(row.dataset.i)];
    const L = layout();
    if (p.start < state.t0 || p.start > state.t0 + visibleDur(L)) state.t0 = p.start - visibleDur(L) * 0.2;
    seek(p.start);
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target.tagName !== "SELECT" && e.target.tagName !== "INPUT") {
      e.preventDefault();
      togglePlay();
    }
  });
  window.addEventListener("resize", resize);
}

function hover(e) {
  const tip = $("tooltip");
  const L = layout();
  const x = e.offsetX;
  const y = e.offsetY;
  if (x < KEYS_W || !state.result) {
    tip.hidden = true;
    return;
  }
  const t = xt(x);
  let html = `<b>${t.toFixed(3)} s</b>`;
  if (y >= L.roll.y && y < L.roll.y + L.roll.h) {
    const rowH = L.roll.h / (state.pitchHi - state.pitchLo + 1);
    const pitch = state.pitchHi - Math.floor((y - L.roll.y) / rowH);
    html += ` · ${noteName(pitch)}`;
    const n = state.notes.find((q) => t >= q.onset && t < q.offset && q.pitch === pitch);
    if (n) {
      html += `<br><span style="color:var(--game)">GAME ${noteName(n.pitch)}</span> ${n.onset.toFixed(2)}–${n.offset.toFixed(2)} s`;
      if (n.phonemes.length) html += `<br>fonemas: ${escapeHtml(n.phonemes.join(" "))}`;
      if (n.words.length) html += `<br>palavras: ${escapeHtml(n.words.join(" "))}`;
    }
    const r = state.reference.find((q) => t >= q.onset && t < q.offset && q.pitch === pitch);
    if (r) html += `<br><span style="color:var(--ref)">Ref. ${noteName(r.pitch)}</span> ${escapeHtml(r.syllable || "")}`;
  } else if (y >= L.phon.y) {
    const p = state.phonemes[phonemeAt(t)];
    if (p) html += `<br>fonema <b>${escapeHtml(p.phoneme)}</b> (${escapeHtml(p.word || "")})<br>${p.start.toFixed(3)}–${p.end.toFixed(3)} s`;
  }
  tip.innerHTML = html;
  tip.hidden = false;
  const left = Math.min(x + 14, canvas.clientWidth - tip.offsetWidth - 6);
  tip.style.left = `${left}px`;
  tip.style.top = `${Math.max(4, y - tip.offsetHeight - 8)}px`;
}

async function main() {
  bind();
  resize();
  state.examples = await fetch("examples/manifest.json").then((r) => r.json());
  const select = $("example");
  select.innerHTML = state.examples.map((ex, i) => `<option value="${i}">${escapeHtml(ex.title)}</option>`).join("");
  select.addEventListener("change", () => loadExample(state.examples[Number(select.value)]));
  await loadExample(state.examples[0]);
}

main().catch((err) => {
  console.error(err);
  $("loading").innerHTML = `<span>Falha ao carregar: ${escapeHtml(err.message)}</span>`;
});
