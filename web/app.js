"use strict";

// ---------- helpers ----------
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMin = (s) => `${Math.round(s / 60)} min`;
const fmtPct = (x, digits = 0) => `${x > 0 ? "+" : ""}${(100 * x).toFixed(digits)}%`;
const hhmm = (iso) => iso.slice(11, 16);
const LEVEL_VAR = { low: "--good", moderate: "--warning", high: "--serious", "very high": "--critical" };
const LEVEL_ICON = { low: "", moderate: "▲", high: "▲▲", "very high": "▲▲▲" };
const LEVEL_WEIGHT = { low: 2, moderate: 2.5, high: 3.5, "very high": 5 };
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const plural = (n, word, many = `${word}s`) => `${n} ${n === 1 ? word : many}`;

const state = {
  meta: null, map: null, layers: {}, places: {}, picks: {}, pickMode: null,
  options: [], selected: 0, lastReq: null, charts: {}, loaded: {}, rerender: {},
};

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 3500);
}

function levelChip(level) {
  return `<span class="level-chip" style="--dot: var(${LEVEL_VAR[level]})" data-icon="${LEVEL_ICON[level]}">${cap(level)} risk</span>`;
}

function table(headers, rows, numeric = []) {
  const th = headers.map((h, i) => `<th class="${numeric.includes(i) ? "num" : ""}">${esc(h)}</th>`).join("");
  const tr = rows.map((r) => `<tr>${r.map((c, i) => `<td class="${numeric.includes(i) ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
}

// ---------- theme ----------
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("rs-theme"); } catch { /* storage unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("rs-theme", next); } catch { /* ignore */ }
    rerenderAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerenderAll);
}

function rerenderAll() {
  Object.values(state.rerender).forEach((fn) => fn());
  if (state.heat) drawHeatmap(state.heat);
  if (state.options.length) drawRoutes();
}

// ---------- charts ----------
function chartBase() {
  Chart.defaults.font.family = css("--font") || "system-ui";
  Chart.defaults.color = css("--ink-2");
  const axis = (title) => ({
    grid: { color: css("--grid"), drawTicks: false },
    border: { color: css("--axis") },
    ticks: { color: css("--muted"), padding: 6 },
    title: title ? { display: true, text: title, color: css("--ink-2") } : undefined,
  });
  return { axis };
}

// Draws short text labels above chosen points (selective direct labels).
const pointNotes = {
  id: "pointNotes",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.notes) return;
    const { ctx } = chart;
    const meta = chart.getDatasetMeta(opts.dataset ?? 0);
    ctx.save();
    ctx.font = `600 11px ${css("--font")}`;
    ctx.fillStyle = css("--ink");
    ctx.textAlign = "center";
    for (const [i, text] of Object.entries(opts.notes)) {
      const p = meta.data[i];
      if (p) ctx.fillText(text, p.x, p.y - 10);
    }
    ctx.restore();
  },
};

// Labels each line at its right end (used when there are <= 4 series).
const endLabels = {
  id: "endLabels",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.enabled) return;
    const { ctx } = chart;
    ctx.save();
    ctx.font = `12px ${css("--font")}`;
    ctx.fillStyle = css("--ink-2");
    chart.data.datasets.forEach((ds, i) => {
      const meta = chart.getDatasetMeta(i);
      if (meta.hidden) return;
      const last = [...meta.data].reverse().find((p, j) => ds.data[ds.data.length - 1 - j] != null);
      if (last) ctx.fillText(ds.label, last.x + 6, last.y + 4);
    });
    ctx.restore();
  },
};
Chart.register(pointNotes, endLabels);

function makeChart(key, canvas, config) {
  if (state.charts[key]) state.charts[key].destroy();
  state.charts[key] = new Chart(canvas, config);
}

// ---------- map ----------
function initMap() {
  const m = state.meta;
  const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView(m.center, 13);
  state.map = map;
  state.renderer = L.canvas({ padding: 0.3 });
  state.layers.base = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "© OpenStreetMap contributors",
  });
  state.layers.risk = L.layerGroup().addTo(map);
  state.layers.hotspots = L.layerGroup();
  state.layers.routes = L.layerGroup().addTo(map);
  state.layers.markers = L.layerGroup().addTo(map);
  const useBase = !m.synthetic_roads;
  $("#layer-base").checked = useBase;
  if (useBase) state.layers.base.addTo(map);

  $("#layer-base").addEventListener("change", (e) => e.target.checked ? state.layers.base.addTo(map) : map.removeLayer(state.layers.base));
  $("#layer-risk").addEventListener("change", (e) => e.target.checked ? state.layers.risk.addTo(map) : map.removeLayer(state.layers.risk));
  $("#layer-moderate").addEventListener("change", loadHeatmap);

  const legend = L.control({ position: "bottomleft" });
  legend.onAdd = () => {
    const div = L.DomUtil.create("div", "legend");
    div.innerHTML = `<div><strong>Road risk</strong> <span class="muted">for the chosen time, weather, vehicle</span></div>` +
      [["very high", "top 5% of fleet km"], ["high", "top 20%"], ["moderate", "above median"]]
        .map(([lvl, txt]) => `<div><span class="sw" style="height:${LEVEL_WEIGHT[lvl]}px;background:var(${LEVEL_VAR[lvl]})"></span>${LEVEL_ICON[lvl]} ${cap(lvl)} <span class="muted">${txt}</span></div>`).join("") +
      `<div><span class="sw" style="height:5px;background:var(--series-1)"></span>Selected route</div>` +
      `<div><span class="sw" style="height:4px;background:var(--route-other)"></span>Alternative route</div>`;
    return div;
  };
  legend.addTo(map);

  map.on("click", (e) => {
    if (!state.pickMode) return;
    const which = state.pickMode;
    state.picks[which] = { lat: e.latlng.lat, lon: e.latlng.lng };
    const sel = $(`#${which}`);
    let opt = sel.querySelector('option[value="__map"]');
    if (!opt) { opt = new Option("Point on map", "__map"); sel.add(opt, 0); }
    sel.value = "__map";
    setPickMode(null);
    drawEndpoints();
  });
}

function setPickMode(which) {
  state.pickMode = which;
  $$("[data-pick]").forEach((b) => b.classList.toggle("active", b.dataset.pick === which));
  $("#pick-hint").hidden = !which;
  $("#map").style.cursor = which ? "crosshair" : "";
}

function pointOf(which) {
  const v = $(`#${which}`).value;
  if (v === "__map") return state.picks[which];
  const p = state.places[v];
  return { lat: p.lat, lon: p.lon };
}

function drawEndpoints() {
  state.layers.markers.clearLayers();
  for (const [which, label] of [["origin", "A"], ["dest", "B"]]) {
    const p = pointOf(which);
    if (!p) continue;
    L.marker([p.lat, p.lon], {
      icon: L.divIcon({ className: "", html: `<div class="hazard-num" style="border-color:var(--series-1)">${label}</div>`, iconSize: [22, 22] }),
      title: which === "origin" ? "Start" : "Destination",
    }).addTo(state.layers.markers);
  }
}

// ---------- risk overlay ----------
function conditionsQuery() {
  return new URLSearchParams({ depart: $("#depart").value, rain: $("#rain").checked, vehicle_type: $("#vehicle").value });
}

async function loadHeatmap() {
  try {
    const minLevel = $("#layer-moderate").checked ? "moderate" : "high";
    state.heat = await api(`/api/heatmap?${conditionsQuery()}&limit=5000&min_level=${encodeURIComponent(minLevel)}`);
    drawHeatmap(state.heat);
  } catch (e) { toast(`Risk overlay: ${e.message}`); }
}

function drawHeatmap(heat) {
  state.layers.risk.clearLayers();
  const ordered = [...heat.segments].reverse(); // draw highest risk last (on top)
  for (const s of ordered) {
    const line = L.polyline(s.coords, {
      renderer: state.renderer, color: css(LEVEL_VAR[s.level]), weight: LEVEL_WEIGHT[s.level], opacity: 0.85,
    });
    line.on("click", () => segmentPopup(s.id, line.getCenter()));
    line.addTo(state.layers.risk);
  }
}

async function segmentPopup(id, latlng) {
  try {
    const d = await api(`/api/segment/${id}?${conditionsQuery()}`);
    const reasons = d.reasons.map((r) => `<li>${esc(r.text)}${r.effect_pct != null ? ` <span class="muted">(${r.effect_pct > 0 ? "+" : ""}${r.effect_pct}%)</span>` : ""}</li>`).join("");
    L.popup().setLatLng(latlng).setContent(
      `<div class="popup-title">${esc(d.name || "Unnamed road")}</div>${levelChip(d.level)}
       <div class="muted">${esc(d.road_class)} · ${d.vehicle_km.toLocaleString()} fleet km · ${plural(d.observed.near_miss, "near-miss", "near-misses")}, ${plural(d.observed.accident, "accident")}</div>
       <ul class="popup-reasons">${reasons || "<li>No single dominant factor</li>"}</ul>`
    ).openOn(state.map);
  } catch (e) { toast(e.message); }
}

// ---------- dispatch: routes ----------
function tripRequest() {
  return {
    origin: pointOf("origin"), dest: pointOf("dest"),
    depart: $("#depart").value, rain: $("#rain").checked, vehicle_type: $("#vehicle").value,
    risk_aversion: Number($('input[name="aversion"]:checked').value),
  };
}

async function findRoutes(e) {
  e.preventDefault();
  const req = tripRequest();
  if (!req.origin || !req.dest) { toast("Choose a start and destination"); return; }
  const btn = $("#find-btn");
  btn.disabled = true;
  btn.textContent = "Finding routes…";
  try {
    const [opt, dep] = await Promise.all([api("/api/route/options", req), api("/api/route/departure", req)]);
    state.lastReq = req;
    state.options = opt.options;
    state.selected = Math.max(0, state.options.findIndex((o) => o.recommended));
    renderOptions();
    drawRoutes(true);
    renderDeparture(dep);
    $("#briefing-card").hidden = false;
    $("#assign-card").hidden = false;
    $("#brief-text").hidden = true;
    $("#brief-source").hidden = true;
    loadHeatmap();
  } catch (err) {
    toast(`Routing failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Find routes";
  }
}

function renderOptions() {
  $("#results").innerHTML = state.options.map((o, i) => {
    const dRisk = o.risk_change_pct;
    const deltas = i === 0 ? "Reference route" :
      `<span class="${dRisk < 0 ? "delta-good" : ""}">${fmtPct(dRisk)} risk</span> · ${o.extra_min >= 0 ? "+" : ""}${o.extra_min.toFixed(1)} min vs fastest`;
    return `<div class="option ${i === state.selected ? "selected" : ""}" role="button" tabindex="0" data-i="${i}" aria-pressed="${i === state.selected}">
      <div class="option-head"><span class="option-label">${esc(o.label)}</span>${o.recommended ? '<span class="rec">Recommended ✓</span>' : ""}
        <span style="margin-left:auto">${levelChip(o.level)}</span></div>
      <div class="option-main"><span class="option-time">${fmtMin(o.duration_s)}</span>
        <span class="option-meta">${o.distance_km.toFixed(1)} km · risk index ${o.risk.toFixed(2)} (${o.risk_lo.toFixed(2)}–${o.risk_hi.toFixed(2)})</span></div>
      <div class="option-meta">${deltas}</div>
      <div class="option-meta">≈ ${o.near_miss_per_1k_trips.toFixed(0)} near-misses and ${o.accidents_per_1k_trips.toFixed(1)} accidents per 1,000 such trips</div>
      ${o.sparse_km_share >= 0.1 ? `<div class="warn-line">${Math.round(100 * o.sparse_km_share)}% of this route is on roads with little fleet data</div>` : ""}
    </div>`;
  }).join("") + `<p class="hint">Risk index = expected severity-weighted events (harsh = 1, near-miss = 10, accident = 100) for this trip, with a 95% interval.</p>`;
  $$("#results .option").forEach((el) => {
    const pick = () => { state.selected = Number(el.dataset.i); renderOptions(); drawRoutes(); };
    el.addEventListener("click", pick);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
}

function drawRoutes(fit = false) {
  const g = state.layers.routes;
  g.clearLayers();
  const order = state.options.map((_, i) => i).filter((i) => i !== state.selected).concat([state.selected]);
  for (const i of order) {
    const o = state.options[i];
    const sel = i === state.selected;
    if (sel) L.polyline(o.geometry, { color: css("--surface"), weight: 10, opacity: 1 }).addTo(g);
    const line = L.polyline(o.geometry, { color: sel ? css("--series-1") : css("--route-other"), weight: sel ? 6 : 4, opacity: sel ? 1 : 0.85 });
    line.on("click", () => { state.selected = i; renderOptions(); drawRoutes(); });
    line.bindTooltip(`${o.label}: ${fmtMin(o.duration_s)}, ${o.level} risk`, { sticky: true });
    line.addTo(g);
  }
  const o = state.options[state.selected];
  [...o.top_segments].slice(0, 4).sort((a, b) => a.km_from_start - b.km_from_start).forEach((t, k) => {
    const reasons = (t.reasons || []).map((r) => `<li>${esc(r.text)}</li>`).join("");
    L.marker([t.lat, t.lon], { icon: L.divIcon({ className: "", html: `<div class="hazard-num">${k + 1}</div>`, iconSize: [22, 22] }) })
      .bindPopup(`<div class="popup-title">${esc(t.name || "Unnamed road")}</div>
        <div class="muted">km ${t.km_from_start.toFixed(1)} · ${Math.round(100 * t.share)}% of this route's risk · ${t.times_typical.toFixed(1)}× a typical fleet km</div>
        <ul class="popup-reasons">${reasons}</ul>`)
      .addTo(g);
  });
  drawEndpoints();
  if (fit) state.map.fitBounds(L.latLngBounds(state.options.flatMap((x) => x.geometry)), { padding: [30, 30] });
}

// ---------- dispatch: departure advisor ----------
function renderDeparture(dep) {
  $("#departure-card").hidden = false;
  $("#departure-advice").textContent = dep.advice;
  const slots = dep.slots;
  const labels = slots.map((s) => hhmm(s.depart));
  const planned = slots.findIndex((s) => s.offset_min === 0);
  const best = slots.findIndex((s) => s.depart === dep.best.depart);
  const notes = { [planned]: "Planned" };
  if (best !== planned) notes[best] = "Best";
  const radius = slots.map((_, i) => (i in notes ? 5 : 0));

  state.rerender.departure = () => {
    const { axis } = chartBase();
    const blue = css("--series-1");
    makeChart("depRisk", $("#dep-risk-chart"), {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "95% upper", data: slots.map((s) => s.risk_hi), borderWidth: 0, pointRadius: 0, fill: "+1", backgroundColor: css("--series-1-wash") },
          { label: "95% lower", data: slots.map((s) => s.risk_lo), borderWidth: 0, pointRadius: 0, fill: false },
          { label: "Route risk index", data: slots.map((s) => s.risk), borderColor: blue, backgroundColor: blue, borderWidth: 2,
            pointRadius: radius, pointHoverRadius: 6, pointBorderColor: css("--surface"), pointBorderWidth: 2, tension: 0.25 },
        ],
      },
      options: {
        maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
        layout: { padding: { top: 16 } },
        plugins: {
          legend: { display: false },
          title: { display: true, text: "Route risk by departure time (with 95% interval)", color: css("--ink"), align: "start" },
          tooltip: {
            filter: (item) => item.datasetIndex === 2,
            callbacks: { label: (item) => { const s = slots[item.dataIndex]; return `Risk ${s.risk.toFixed(2)} (${s.risk_lo.toFixed(2)}–${s.risk_hi.toFixed(2)}) · ${s.label} route`; } },
          },
          pointNotes: { dataset: 2, notes },
        },
        scales: { x: axis(), y: axis("Risk index") },
      },
    });
    makeChart("depTime", $("#dep-time-chart"), {
      type: "line",
      data: { labels, datasets: [{ label: "Travel time (min)", data: slots.map((s) => s.duration_s / 60), borderColor: blue, backgroundColor: blue,
        borderWidth: 2, pointRadius: radius, pointBorderColor: css("--surface"), pointBorderWidth: 2, tension: 0.25 }] },
      options: {
        maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false }, title: { display: true, text: "Travel time (min)", color: css("--ink"), align: "start" },
          tooltip: { callbacks: { label: (item) => `${item.parsed.y.toFixed(1)} min` } } },
        scales: { x: axis(), y: axis() },
      },
    });
  };
  state.rerender.departure();
  $("#dep-table").innerHTML = table(["Depart", "Arrive", "Minutes", "Risk", "95% interval", "Route"],
    slots.map((s) => [hhmm(s.depart), hhmm(s.arrive), (s.duration_s / 60).toFixed(1), s.risk.toFixed(2), `${s.risk_lo.toFixed(2)}–${s.risk_hi.toFixed(2)}`, s.label]), [2, 3]);
}

// ---------- dispatch: briefing, assignment, log ----------
async function makeBriefing() {
  const o = state.options[state.selected];
  const r = state.lastReq;
  const btn = $("#brief-btn");
  btn.disabled = true;
  try {
    const b = await api("/api/briefing", { segment_ids: o.segment_ids, depart: r.depart, rain: r.rain, vehicle_type: r.vehicle_type, language: $("#lang").value });
    $("#brief-text").textContent = b.text;
    $("#brief-text").hidden = false;
    const src = $("#brief-source");
    src.textContent = b.source === "llm" ? "Written by AI (gpt-oss on Groq)" : "Template (AI briefing unavailable)";
    src.hidden = false;
  } catch (e) { toast(`Briefing failed: ${e.message}`); } finally { btn.disabled = false; }
}

async function assignTrip() {
  const o = state.options[state.selected];
  const f = state.options[0];
  const rec = state.options.find((x) => x.recommended);
  const r = state.lastReq;
  try {
    const { id } = await api("/api/dispatch", {
      ...r, segment_ids: o.segment_ids, option_label: o.label, recommended_label: rec.label,
      followed_recommendation: o === rec, risk: o.risk, risk_lo: o.risk_lo, risk_hi: o.risk_hi, level: o.level,
      duration_s: o.duration_s, distance_km: o.distance_km, fastest_risk: f.risk, fastest_duration_s: f.duration_s,
      near_miss_per_1k_trips: o.near_miss_per_1k_trips, accidents_per_1k_trips: o.accidents_per_1k_trips,
      driver: $("#driver").value || null, vehicle: $("#vehicle-id").value || null,
    });
    toast(`Trip #${id} assigned (${o.label} route, ${o.level} risk)`);
    loadDispatchLog();
    state.loaded.analytics = false;
  } catch (e) { toast(`Could not assign: ${e.message}`); }
}

async function loadDispatchLog() {
  const rows = await api("/api/dispatch?limit=5");
  $("#dispatch-log").innerHTML = rows.length ? rows.map((d) => `
    <div class="log-item">
      <div><strong>#${d.id}</strong> ${esc(d.option_label)} route · ${esc(d.level)} risk · ${fmtMin(d.duration_s)}
        ${d.driver ? `· ${esc(d.driver)}` : ""} <span class="muted">${esc(d.created_at.replace("T", " "))}</span></div>
      ${d.outcome ? `<div class="muted">Reported: ${d.outcome.harsh_events} harsh, ${d.outcome.near_misses} near-misses, ${d.outcome.accidents} accidents</div>` :
        `<div class="row" data-id="${d.id}"><span class="muted">Post-trip report:</span>
          <label>Harsh <input type="number" min="0" value="0"></label>
          <label>Near-miss <input type="number" min="0" value="0"></label>
          <label>Accident <input type="number" min="0" value="0"></label>
          <button class="ghost-btn">Save</button></div>`}
    </div>`).join("") : `<p class="hint">No trips assigned yet.</p>`;
  $$("#dispatch-log .row[data-id] button").forEach((b) => b.addEventListener("click", async () => {
    const row = b.parentElement;
    const [h, n, a] = $$("input", row).map((i) => Number(i.value) || 0);
    try {
      await api(`/api/dispatch/${row.dataset.id}/outcome`, { harsh_events: h, near_misses: n, accidents: a });
      loadDispatchLog();
    } catch (e) { toast(e.message); }
  }));
}

// ---------- hotspots & alerts ----------
async function loadHotspotsTab() {
  const [alerts, hot] = await Promise.all([api("/api/alerts"), api(`/api/hotspots?limit=500`)]);
  state.alerts = alerts;
  state.hotspots = hot.items;
  $("#alert-count").textContent = `(${alerts.length})`;
  $("#alert-list").innerHTML = alerts.length ? alerts.map((a, i) => `
    <button class="list-item" data-alert="${i}">
      <span class="title"><span class="new-tag">NEW</span>${esc(cap(a.headline))}</span>
      <span class="meta">First flagged ${esc((a.first_flagged || "").slice(0, 10))} · ${a.observed} events vs ${a.expected.toFixed(1)} expected</span>
    </button>`).join("") : `<p class="hint">No roads with a significant recent rise.</p>`;
  $$("[data-alert]").forEach((el) => el.addEventListener("click", () => focusPoint(alerts[el.dataset.alert], "alert")));
  renderHotspotList();
  drawHotspots();
}

function renderHotspotList() {
  const q = $("#hotspot-q").value.trim().toLowerCase();
  const kind = $("#hotspot-kind").value;
  const items = state.hotspots.filter((h) => (!kind || h.kind === kind) &&
    (!q || String(h.hotspot_id) === q || (h.name || "").toLowerCase().includes(q)));
  $("#hotspot-count").textContent = `(${items.length} of ${state.hotspots.length})`;
  $("#hotspot-list").innerHTML = items.slice(0, 200).map((h) => `
    <button class="list-item" data-hot="${h.hotspot_id}">
      <span class="title">#${h.hotspot_id + 1} ${esc(h.name || "Unnamed")} <span class="muted">${esc(h.kind)}</span></span>
      <span class="meta">${h.excess_ratio.toFixed(1)}× typical · ${plural(h.n_near_miss, "near-miss", "near-misses")} · ${plural(h.n_accident, "accident")} · ${h.n_harsh} harsh
        · 30-day trend ${h.trend_30d >= 1.2 ? "rising" : h.trend_30d <= 0.8 ? "falling" : "steady"}</span>
    </button>`).join("");
  $$("[data-hot]").forEach((el) => el.addEventListener("click", () =>
    focusPoint(state.hotspots.find((h) => h.hotspot_id === Number(el.dataset.hot)), "hotspot")));
}

function hotspotPopup(h) {
  return `<div class="popup-title">#${h.hotspot_id + 1} ${esc(h.name || "Unnamed")} (${esc(h.kind)})</div>
    <div>${h.excess_ratio.toFixed(1)}× the typical rate for similar ${h.kind === "intersection" ? "junctions" : "roads"}</div>
    <div class="muted">${h.n_events} events: ${h.n_harsh} harsh, ${plural(h.n_near_miss, "near-miss", "near-misses")}, ${plural(h.n_accident, "accident")} · last ${esc((h.last_event_ts || "").slice(0, 10))}</div>`;
}

function alertPopup(a) {
  return `<div class="popup-title"><span class="new-tag">NEW</span>${esc(cap(a.headline))}</div>
    <div class="muted">${a.observed} events in 14 days vs ${a.expected.toFixed(1)} expected · first flagged ${esc((a.first_flagged || "").slice(0, 10))}</div>`;
}

function drawHotspots() {
  const g = state.layers.hotspots;
  g.clearLayers();
  for (const h of state.hotspots) {
    L.circleMarker([h.lat, h.lon], {
      radius: Math.min(16, 4 + Math.sqrt(Math.max(h.excess_events, 0)) / 1.5), color: css("--serious"), weight: 2,
      fillColor: css("--serious"), fillOpacity: 0.3,
    }).bindPopup(hotspotPopup(h)).addTo(g);
  }
  for (const a of state.alerts) {
    L.marker([a.lat, a.lon], { icon: L.divIcon({ className: "", html: `<div class="hazard-num" style="border-color:var(--critical);width:34px;border-radius:6px">NEW</div>`, iconSize: [34, 22] }) })
      .bindPopup(alertPopup(a)).addTo(g);
  }
}

function focusPoint(item, kind) {
  state.map.flyTo([item.lat, item.lon], 16, { duration: 0.6 });
  L.popup().setLatLng([item.lat, item.lon]).setContent(kind === "alert" ? alertPopup(item) : hotspotPopup(item)).openOn(state.map);
}

// ---------- analytics ----------
function tile(label, value, sub = "") {
  return `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

async function loadAnalytics() {
  const [s, d] = await Promise.all([api("/api/analytics/summary"), api("/api/dispatch/summary")]);
  const change = s.serious_per_100km_prev_4_weeks ? s.serious_per_100km_last_4_weeks / s.serious_per_100km_prev_4_weeks - 1 : null;
  $("#kpis").innerHTML = [
    tile("Fleet distance", `${Math.round(s.vehicle_km).toLocaleString()} km`),
    tile("Harsh driving events", s.harsh_events.toLocaleString()),
    tile("Near-misses", s.near_misses.toLocaleString()),
    tile("Accidents", s.accidents.toLocaleString()),
    tile("Serious events / 100 km", s.serious_per_100km_last_4_weeks.toFixed(2),
      change == null ? "last 4 weeks" : `last 4 weeks · ${fmtPct(change, 1)} vs previous 4`),
  ].join("");
  $("#dispatch-summary").innerHTML = d.dispatches ? [
    tile("Trips assigned", d.dispatches),
    tile("Followed recommendation", `${d.followed_recommendation_pct}%`),
    tile("Avg risk vs fastest", `${d.avg_risk_change_vs_fastest_pct > 0 ? "+" : ""}${d.avg_risk_change_vs_fastest_pct}%`),
    tile("Avg extra time", `${d.avg_extra_min} min`),
    tile("Completed trips", d.completed, `${d.completed_reported_serious} serious events reported vs ${d.completed_predicted_serious} predicted`),
  ].join("") : `<p class="hint">No trips assigned yet. Assign one from the Dispatch tab.</p>`;
  await loadTrends();
}

const MAX_SERIES = 7;

async function loadTrends() {
  const dim = $("#dim").value;
  const metric = $("#metric").value;
  const raw = await api(`/api/analytics/trends?dimension=${dim}&metric=${metric}`);
  // Keep the 7 biggest groups by distance (stable order), fold the rest into "Other".
  const groups = Object.entries(raw).map(([g, s]) => [g, s, s.reduce((a, x) => a + x.km, 0)]).sort((a, b) => b[2] - a[2]);
  const weeks = [...new Set(Object.values(raw).flatMap((s) => s.map((x) => x.week)))].sort();
  const totalKm = groups.reduce((a, g) => a + g[2], 0);
  // Keep the biggest groups (by distance, a stable order); fold small ones into "Other".
  const big = groups.filter((g) => g[2] >= 0.02 * totalKm);
  const keep = big.slice(0, big.length > MAX_SERIES + 1 ? MAX_SERIES : big.length);
  const rest = groups.filter((g) => !keep.includes(g));
  const series = keep.map(([g, s]) => ({ label: g, values: weeks.map((w) => s.find((x) => x.week === w)?.value ?? null) }));
  if (rest.length) {
    series.push({
      label: `Other (${rest.length})`,
      values: weeks.map((w) => {
        let ev = 0, km = 0;
        for (const [, s] of rest) { const x = s.find((p) => p.week === w); if (x) { ev += x.events; km += x.km; } }
        return km >= 500 ? 100 * ev / km : null;
      }),
    });
  }
  const metricName = $("#metric").selectedOptions[0].text;
  $("#trend-title").textContent = `${metricName} per 100 km, by week`;
  state.rerender.trends = () => {
    const { axis } = chartBase();
    makeChart("trends", $("#trend-chart"), {
      type: "line",
      data: {
        labels: weeks,
        datasets: series.map((s, i) => {
          const color = css(`--series-${i + 1}`);
          return { label: s.label, data: s.values, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, tension: 0.25, spanGaps: true };
        }),
      },
      options: {
        maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
        layout: { padding: { right: series.length <= 4 ? 90 : 8 } },
        plugins: {
          legend: { position: "top", align: "start", labels: { boxWidth: 12, boxHeight: 2, color: css("--ink-2") } },
          tooltip: { callbacks: { label: (it) => `${it.dataset.label}: ${it.parsed.y == null ? "–" : it.parsed.y.toFixed(3)}` } },
          endLabels: { enabled: series.length <= 4 },
        },
        scales: { x: axis("Week starting"), y: axis("per 100 km") },
      },
    });
  };
  state.rerender.trends();
  $("#trend-table").innerHTML = table(["Week", ...series.map((s) => s.label)],
    weeks.map((w, i) => [w, ...series.map((s) => (s.values[i] == null ? "–" : s.values[i].toFixed(3)))]),
    series.map((_, i) => i + 1));
}

// ---------- evaluation ----------
async function loadEvaluation() {
  let ev;
  try { ev = await api("/api/eval"); } catch { $("#eval-empty").hidden = false; return; }
  const tb = ev.trip_backtest, sb = ev.segment_backtest, im = ev.impact, em = ev.emerging_detection;
  $("#eval-tiles").innerHTML = [
    tile("Serious events avoided", `${im.true_serious_events_cut_pct_all_trips.toFixed(1)}%`,
      `${im.serious_events_avoided_per_10k_trips.toFixed(0)} per 10,000 trips · ${im.true_serious_events_cut_pct_rerouted_trips.toFixed(1)}% on rerouted trips`),
    tile("Extra travel time", `${im.extra_time_pct_all_trips.toFixed(1)}%`, `${im.extra_minutes_per_rerouted_trip.toFixed(1)} min per rerouted trip · ${im.rerouted_pct}% rerouted`),
    tile("Trip risk AUC", tb.auc_full_model.toFixed(3), `distance alone ${tb.auc_distance_only.toFixed(3)}`),
    tile("Riskiest 10% of km", `${sb.full_model.top10pct_km_lift.toFixed(2)}×`, "the average serious-event rate"),
    tile("New hotspots caught", `${em.detected} of ${em.hazards}`, `median ${em.median_delay_days} days · ${em.false_alarms_per_week} false alarms / week`),
    tile("Route options", `${ev.timings.route_options_ms_per_trip} ms`, "per request, all alternatives"),
  ].join("");
  const names = { distance_only: "Distance only", raw_history: "Raw history", features_only: "Road features only", full_model: "Full model" };
  $("#eval-methods").innerHTML = table(["Method", "Deviance explained", "Top-10% km lift"],
    Object.keys(names).map((k) => [names[k], sb[k].deviance_explained.toFixed(3), `${sb[k].top10pct_km_lift.toFixed(2)}×`]), [1, 2]);
  const cal = tb.calibration_by_decile;
  state.rerender.calib = () => {
    const { axis } = chartBase();
    const maxv = Math.max(...cal.predicted, ...cal.observed) * 1.1;
    makeChart("calib", $("#calib-chart"), {
      type: "scatter",
      data: {
        datasets: [
          { label: "Perfect calibration", type: "line", data: [{ x: 0, y: 0 }, { x: maxv, y: maxv }], borderColor: css("--muted"), backgroundColor: css("--muted"), borderWidth: 1, pointRadius: 0 },
          { label: "Test trips, by decile of predicted risk", data: cal.predicted.map((p, i) => ({ x: p, y: cal.observed[i] })),
            backgroundColor: css("--series-1"), borderColor: css("--surface"), borderWidth: 2, pointRadius: 6, pointHoverRadius: 8 },
        ],
      },
      options: {
        maintainAspectRatio: false, animation: false,
        plugins: { legend: { position: "top", align: "start", labels: { boxWidth: 12, color: css("--ink-2") } },
          tooltip: { callbacks: { label: (it) => `predicted ${it.parsed.x.toFixed(3)}, observed ${it.parsed.y.toFixed(3)}` } } },
        scales: { x: { ...axis("Predicted serious events per trip"), min: 0, max: maxv }, y: { ...axis("Observed"), min: 0, max: maxv } },
      },
    });
  };
  state.rerender.calib();
  $("#calib-table").innerHTML = table(["Decile", "Predicted", "Observed"], cal.predicted.map((p, i) => [i + 1, p.toFixed(4), cal.observed[i].toFixed(4)]), [1, 2]);
  const d = ev.data;
  $("#eval-notes").innerHTML = `
    <div>Trained on data before <strong>${esc(ev.split)}</strong>, tested after: ${d.train_vehicle_km.toLocaleString()} train and ${d.test_vehicle_km.toLocaleString()} test vehicle-km, ${d.trips.toLocaleString()} trips.</div>
    <div>Impact compares the recommended and fastest routes on ${im.trips_evaluated.toLocaleString()} sampled test trips. ${esc(im.note)}</div>
    <div>Agreement with the simulator's hidden per-road risk (Spearman): near-miss ${ev.truth_recovery.near_miss.full_model} (raw history ${ev.truth_recovery.near_miss.raw_history}), harsh ${ev.truth_recovery.harsh.full_model}.</div>
    ${state.meta.synthetic_roads ? "<div><strong>These numbers come from the synthetic development city.</strong> Re-run the evaluation on the Bengaluru data for figures to present.</div>" : ""}`;
}

// ---------- tabs ----------
function showTab(tab) {
  $$(".tab").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  $$("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== tab; });
  $("#layout").classList.toggle("no-map", tab === "analytics" || tab === "evaluation");
  const map = state.map;
  map.closePopup();
  if (tab === "hotspots") { state.layers.hotspots.addTo(map); map.removeLayer(state.layers.routes); }
  else { map.removeLayer(state.layers.hotspots); state.layers.routes.addTo(map); }
  setTimeout(() => map.invalidateSize(), 50);
  const loaders = { hotspots: loadHotspotsTab, analytics: loadAnalytics, evaluation: loadEvaluation };
  if (loaders[tab] && !state.loaded[tab]) {
    state.loaded[tab] = true;
    loaders[tab]().catch((e) => { state.loaded[tab] = false; toast(e.message); });
  }
}

// ---------- init ----------
async function init() {
  initTheme();
  const m = await api("/api/meta");
  state.meta = m;
  const badge = $("#data-badge");
  badge.hidden = false;
  badge.textContent = m.synthetic_roads ? "Synthetic dev city" : `Region: ${m.region}`;
  if (m.synthetic_roads) { badge.classList.add("warn"); $("#synthetic-banner").hidden = false; }

  for (const p of m.places) state.places[p.id] = p;
  const opts = (kind) => m.places.filter((p) => p.kind === kind).map((p) => `<option value="${p.id}">${esc(p.label)}</option>`).join("");
  const html = `<optgroup label="Depots">${opts("depot")}</optgroup><optgroup label="Customers">${opts("customer")}</optgroup>`;
  $("#origin").innerHTML = html;
  $("#dest").innerHTML = html;
  $("#origin").value = m.places.find((p) => p.kind === "depot").id;
  // Demo default: the customer farthest from the first depot (a trip with real alternatives).
  const depot = m.places.find((p) => p.kind === "depot");
  const far = m.places.filter((p) => p.kind === "customer")
    .sort((a, b) => ((b.lat - depot.lat) ** 2 + (b.lon - depot.lon) ** 2) - ((a.lat - depot.lat) ** 2 + (a.lon - depot.lon) ** 2))[0];
  $("#dest").value = (far ?? m.places[1]).id;
  $("#depart").value = `${m.as_of.slice(0, 10)}T18:15`;

  initMap();
  drawEndpoints();
  loadHeatmap();
  loadDispatchLog().catch((e) => toast(e.message));

  $("#trip-form").addEventListener("submit", findRoutes);
  $$("[data-pick]").forEach((b) => b.addEventListener("click", () => setPickMode(state.pickMode === b.dataset.pick ? null : b.dataset.pick)));
  ["origin", "dest"].forEach((id) => $(`#${id}`).addEventListener("change", drawEndpoints));
  ["depart", "rain", "vehicle"].forEach((id) => $(`#${id}`).addEventListener("change", loadHeatmap));
  $("#brief-btn").addEventListener("click", makeBriefing);
  $("#assign-btn").addEventListener("click", assignTrip);
  $("#hotspot-q").addEventListener("input", renderHotspotList);
  $("#hotspot-kind").addEventListener("change", renderHotspotList);
  $("#dim").addEventListener("change", () => loadTrends().catch((e) => toast(e.message)));
  $("#metric").addEventListener("change", () => loadTrends().catch((e) => toast(e.message)));
  $$(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  if (!m.llm_briefings) $("#lang").title = "Hindi and Kannada need a Groq API key on the server";
}

init().catch((e) => toast(`Failed to start: ${e.message}`));
