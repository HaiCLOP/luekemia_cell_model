"use strict";

const POLL_MS = 3000;
const RUN_COLORS = ["#6b8fe8", "#f0907a", "#5cc28f", "#e9b24f", "#59c3c8", "#d98bb8", "#a3b86c", "#c99a6e", "#8aa0b8", "#e26363", "#9fd3a8", "#f2c38b"];
const gpuHistory = [];
let lastState = null;
let metric = "pr_auc";

const $ = (id) => document.getElementById(id);
const isNum = (v) => typeof v === "number" && !Number.isNaN(v);
const f = (v, d = 3) => (isNum(v) ? v.toFixed(d) : "–");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function dur(min) {
  if (!isNum(min)) return "–";
  const m = Math.round(min);
  if (m < 1) return "<1 min";
  const h = Math.floor(m / 60), r = m % 60;
  return h ? `${h}h ${String(r).padStart(2, "0")}m` : `${r} min`;
}
const hhmm = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

/* ================= header ================= */
function tickClock() { $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false }); }
setInterval(tickClock, 1000); tickClock();

function renderHeader(s) {
  const running = !!s.current;
  const failed = (s.timeline || []).some((t) => t.status === "failed");
  $("pulse").className = "pulse " + (running ? "on" : failed ? "fail" : "");
  $("state-text").textContent = running ? "Training" : s.pending ? "Paused / waiting" : "Queue complete";
  if (s.queue_eta && running) {
    const eta = new Date(s.queue_eta);
    const mins = (eta - new Date()) / 60000;
    $("queue-eta").innerHTML = `Queue finishes ≈ <b>${hhmm(eta)}</b> · in ${dur(mins)} · ${s.pending} job${s.pending === 1 ? "" : "s"} after this one`;
  } else {
    $("queue-eta").textContent = running ? "estimating queue finish" : "no job running";
  }
}

/* ================= now training ================= */
const C_STAGE = 2 * Math.PI * 50, C_BATCH = 2 * Math.PI * 40;
function setRing(id, circ, frac) {
  const el = $(id);
  el.style.strokeDasharray = `${circ}`;
  el.style.strokeDashoffset = `${circ * (1 - clamp(frac || 0, 0, 1))}`;
}

function renderNow(s) {
  const d = s.detail || {}, ep = s.epochs || [];
  const running = !!s.current;
  $("run-label").textContent = s.focus_label || "No run yet";
  $("run-id").textContent = s.focus || "–";

  // stepper
  const st = d.stage;
  const steps = [...$("stepper").children];
  if (st === "scratch") {
    steps[0].querySelector("div").innerHTML = "<b>From scratch</b><small>all layers train from random init</small>";
    steps[1].style.display = "none";
  } else {
    steps[0].querySelector("div").innerHTML = "<b>Stage 1</b><small>head only, backbone frozen</small>";
    steps[1].style.display = "";
  }
  const order = st === "scratch" ? ["stage1", "saved"] : ["stage1", "stage2", "saved"];
  const activeIdx = !running ? order.length : st === "stage2" ? 1 : 0;
  steps.forEach((li) => {
    const idx = order.indexOf(li.dataset.stage);
    li.className = idx < 0 ? "" : idx < activeIdx ? "done" : idx === activeIdx ? "active" : "";
  });

  // ring
  const stageEp = d.stage_epoch ?? (d.stage_epochs_done ?? 0);
  const stageTot = d.stage_total ?? ({ stage1: 5, stage2: 25, scratch: 30 }[st] || 1);
  const batchFrac = d.batches ? d.batch / d.batches : 0;
  setRing("ring-stage", C_STAGE, running ? (stageEp - 1 + batchFrac) / stageTot : 1);
  setRing("ring-batch", C_BATCH, running ? batchFrac : 1);
  $("ring-epoch").textContent = running && stageEp ? `${stageEp}/${stageTot}` : ep.length ? "done" : "–";
  $("ring-sub").textContent = running ? `epoch · ${st || ""}` : "finished";

  // KPIs
  let bestI = -1;
  ep.forEach((r, i) => { if (isNum(r.val_pr_auc) && (bestI < 0 || r.val_pr_auc > ep[bestI].val_pr_auc)) bestI = i; });
  $("k-best").textContent = bestI >= 0 ? f(ep[bestI].val_pr_auc, 4) : "–";
  $("k-best-note").textContent = bestI >= 0 ? `epoch ${bestI + 1} (${ep[bestI].stage}) · ROC ${f(ep[bestI].val_roc_auc, 3)}` : "after first epoch";
  const last = ep[ep.length - 1], prev = ep[ep.length - 2];
  $("k-last").textContent = last ? f(last.val_pr_auc, 4) : "–";
  if (last && prev && isNum(last.val_pr_auc) && isNum(prev.val_pr_auc)) {
    const dlt = last.val_pr_auc - prev.val_pr_auc;
    $("k-last-note").innerHTML = `<span class="${dlt >= 0 ? "up" : "down"}">${dlt >= 0 ? "▲" : "▼"} ${Math.abs(dlt).toFixed(4)}</span> vs previous · loss ${f(last.train_loss, 3)}`;
  } else $("k-last-note").textContent = last ? `loss ${f(last.train_loss, 3)}` : " ";

  const lim = d.patience_limit || 5, used = clamp(d.since_best || 0, 0, lim);
  $("k-patience").innerHTML = Array.from({ length: lim }, (_, i) => `<i class="${i < used ? "used" + (i === lim - 1 ? " last" : "") : ""}"></i>`).join("");
  $("k-patience").setAttribute("aria-label", `${used} of ${lim} epochs without improvement`);
  $("k-patience-note").textContent = !ep.length ? "starts after first epoch" : used === 0 ? "just improved" : `${used} of ${lim} epochs without improvement`;

  if (running && isNum(d.likely_remaining_min)) {
    $("k-eta").textContent = dur(d.likely_remaining_min);
    $("k-eta-note").textContent = `worst case ${dur(d.max_remaining_min)} · ${f(d.sec_per_epoch, 0)} s/epoch`;
  } else { $("k-eta").textContent = running ? "…" : "0"; $("k-eta-note").textContent = running ? "after first epoch" : "run finished"; }

  // batch line
  const fresh = running && d.batches && d.age_sec < 120;
  $("bl-fill").style.width = `${fresh ? 100 * batchFrac : running ? 0 : 100}%`;
  $("bl-fill").className = "fill" + (fresh ? " live" : "");
  $("bl-track").setAttribute("aria-valuenow", fresh ? Math.round(100 * batchFrac) : 0);
  $("bl-stats").textContent = fresh
    ? `batch ${d.batch}/${d.batches} · ${f(d.img_per_sec, 0)} img/s · loss ${f(d.live_loss, 4)} · lr ${d.lr ? d.lr.toExponential(1) : "–"}`
    : running ? "per-epoch updates only for this run" : "finished";
}

/* ================= GPU ================= */
function arcGauge(svg, frac, value, sub, color) {
  const R = 38, cx = 50, cy = 52, L = Math.PI * R;
  const p = clamp(frac, 0, 1);
  svg.innerHTML = `
    <path class="g-track" d="M${cx - R},${cy} A${R},${R} 0 0 1 ${cx + R},${cy}"/>
    <path class="g-arc" d="M${cx - R},${cy} A${R},${R} 0 0 1 ${cx + R},${cy}" stroke="${color}" stroke-dasharray="${L}" stroke-dashoffset="${L * (1 - p)}"/>
    <text class="g-val" x="${cx}" y="${cy - 6}" text-anchor="middle">${value}</text>
    <text class="g-sub" x="${cx}" y="${cy + 9}" text-anchor="middle">${sub}</text>`;
}
function renderGpu(g, d) {
  if (!g) { $("gpu-name").textContent = "nvidia-smi unavailable"; return; }
  $("gpu-name").textContent = (g.name || "").replace("NVIDIA GeForce ", "");
  arcGauge($("g-util"), g.util / 100, `${f(g.util, 0)}%`, "busy", "var(--hx)");
  arcGauge($("g-mem"), g.mem_used / g.mem_total, `${(g.mem_used / 1024).toFixed(1)}G`, `of ${(g.mem_total / 1024).toFixed(0)} GB`, "var(--eo)");
  const tcol = g.temp >= 85 ? "var(--bad)" : g.temp >= 75 ? "var(--warn)" : "var(--ok)";
  arcGauge($("g-temp"), (g.temp - 30) / 60, `${f(g.temp, 0)}°`, "C", tcol);
  $("g-clock").textContent = isNum(g.clock) ? `${f(g.clock, 0)} MHz` : "–";
  $("g-power").textContent = isNum(g.power) ? `${f(g.power, 1)} W` : "–";
  $("g-ips").textContent = d && isNum(d.img_per_sec) && d.age_sec < 120 ? `${f(d.img_per_sec, 0)} img/s` : "–";
  $("gpu-hint").textContent = g.temp >= 85 ? "Hot. Check airflow: the card may throttle." : g.temp >= 75 ? "Warm. Normal under load, keep vents clear."
    : g.util < 50 && d && d.age_sec < 120 ? "GPU waiting on data loading or validation, which is normal between batches." : " ";
  gpuHistory.push({ util: g.util, temp: g.temp });
  if (gpuHistory.length > 120) gpuHistory.shift();
  const W = 300, H = 70, n = gpuHistory.length;
  if (n < 2) return;
  const x = (i) => (i * W) / (n - 1);
  const path = (k, lo, hi) => gpuHistory.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${(H - 4 - clamp(((p[k] ?? lo) - lo) / (hi - lo), 0, 1) * (H - 10)).toFixed(1)}`).join("");
  const u = path("util", 0, 100);
  $("gpu-spark").innerHTML = `<path d="${u}L${W},${H}L0,${H}Z" fill="var(--hx-soft)"/><path d="${u}" fill="none" stroke="var(--hx)" stroke-width="1.6"/>
    <path d="${path("temp", 30, 90)}" fill="none" stroke="var(--eo)" stroke-width="1.6"/>`;
}

/* ================= timeline ================= */
function renderTimeline(s) {
  const items = s.timeline || [];
  const w = (t) => t.minutes ?? t.est ?? 20;
  const total = items.reduce((a, t) => a + w(t), 0) || 1;
  $("timeline").innerHTML = items.map((t) => {
    const pct = (100 * w(t)) / total;
    const prog = t.status === "running" && t.est ? `<div class="seg-prog" style="width:${clamp(100 * (t.elapsed || 0) / t.est, 0, 100)}%"></div>` : "";
    const short = t.label.replace("Ablation: ", "").replace("EfficientNet-B0 · seed", "B0 s").replace("EfficientNet-B0", "B0");
    return `<div class="seg ${t.status}" style="flex:${pct} 0 0" title="${esc(t.label)} · ${esc(t.status)}">${prog}<span>${pct > 4 ? esc(short) : ""}</span></div>`;
  }).join("");
  const doneMin = items.filter((t) => t.status === "done" || t.status === "failed").reduce((a, t) => a + (t.minutes || 0), 0);
  const left = items.filter((t) => t.status !== "done" && t.status !== "failed").reduce((a, t) => a + Math.max(0, (t.est || 0) - (t.elapsed || 0)), 0);
  $("tl-axis").innerHTML = `<span>${dur(doneMin)} done</span><span>${items.filter((t) => t.status === "done").length} / ${items.length} jobs</span><span>≈ ${dur(left)} left</span>`;
  $("q-summary").textContent = `${items.length} jobs · widths proportional to measured or estimated minutes`;
  $("jobs").innerHTML = items.map((t) => {
    const right = t.status === "done" ? (isNum(t.pr_auc) ? `PR ${t.pr_auc.toFixed(3)}` : "done")
      : t.status === "running" ? "running" : t.status === "failed" ? "failed" : "queued";
    const meta = t.status === "done" ? dur(t.minutes) : t.status === "running" ? `${dur(t.elapsed)} of ≈${dur(t.est)}` : t.status === "pending" ? `≈ ${dur(t.est)}` : "";
    return `<li class="${t.status}"><span class="jd"></span><span class="jl" title="${esc(t.label)}">${esc(t.label)}</span><span class="jr">${right}</span>
      <span class="jm"><span>${esc(t.name)}</span><span>${meta}</span></span></li>`;
  }).join("");
}

/* ================= charts ================= */
function axes(W, H, m, lo, hi, n, opts = {}) {
  const x = (i) => m.l + (n <= 1 ? (W - m.l - m.r) / 2 : (i * (W - m.l - m.r)) / (n - 1));
  const y = (v) => m.t + (1 - (v - lo) / (hi - lo || 1)) * (H - m.t - m.b);
  let g = `<g class="ax">`;
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(opts.dec ?? 2)}</text>`;
  }
  const step = Math.max(1, Math.ceil(n / 10));
  for (let i = 0; i < n; i += step) g += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${i + 1}</text>`;
  g += `<text x="${m.l - 8}" y="${H - 8}" text-anchor="end" style="fill:var(--muted)">ep</text></g>`;
  return { x, y, g };
}
const pathOf = (vals, x, y) => vals.map((v, i) => (isNum(v) ? `${x(i).toFixed(1)},${y(v).toFixed(1)}` : null)).filter(Boolean).map((p, i) => (i ? "L" : "M") + p).join("");

function chartRun(svg, ep) {
  const W = 640, H = 260, m = { l: 48, r: 48, t: 14, b: 28 };
  if (!ep.length) { svg.innerHTML = `<text class="empty" x="${W / 2}" y="${H / 2}" text-anchor="middle">Waiting for the first validation epoch</text>`; return; }
  const aucs = ep.flatMap((r) => [r.val_pr_auc, r.val_roc_auc]).filter(isNum);
  let lo = Math.min(...aucs), hi = Math.max(...aucs);
  const pad = Math.max(0.01, (hi - lo) * 0.15); lo = Math.max(0, lo - pad); hi = Math.min(1, hi + pad);
  const n = ep.length;
  const { x, y, g } = axes(W, H, m, lo, hi, n);
  const losses = ep.map((r) => r.train_loss).filter(isNum);
  const llo = Math.min(...losses) * 0.9, lhi = Math.max(...losses) * 1.05;
  const yl = (v) => m.t + (1 - (v - llo) / (lhi - llo || 1)) * (H - m.t - m.b);
  let out = g;
  for (let k = 0; k <= 4; k++) { const v = llo + ((lhi - llo) * k) / 4; out += `<text x="${W - m.r + 8}" y="${yl(v) + 4}" class="ax" style="fill:var(--muted);font-size:11px">${v.toFixed(2)}</text>`; }
  ep.forEach((r, i) => { if (i && r.stage !== ep[i - 1].stage) {
    const xx = (x(i - 1) + x(i)) / 2;
    out += `<line class="stage-line" x1="${xx}" x2="${xx}" y1="${m.t}" y2="${H - m.b}"/><text class="stage-lbl" x="${xx + 5}" y="${m.t + 10}">${esc(r.stage)}</text>`;
  } });
  const pr = ep.map((r) => r.val_pr_auc), roc = ep.map((r) => r.val_roc_auc);
  const prPath = pathOf(pr, x, y);
  out += `<path class="area eo" d="${prPath}L${x(n - 1)},${H - m.b}L${x(0)},${H - m.b}Z"/>`;
  out += `<path class="ln ink" d="${pathOf(ep.map((r) => r.train_loss), x, yl)}"/>`;
  out += `<path class="ln hx" d="${pathOf(roc, x, y)}"/><path class="ln eo" d="${prPath}"/>`;
  ep.forEach((r, i) => {
    if (isNum(r.val_roc_auc)) out += `<circle class="pt hx" cx="${x(i)}" cy="${y(r.val_roc_auc)}" r="2.6"><title>epoch ${i + 1} ROC-AUC ${r.val_roc_auc.toFixed(4)}</title></circle>`;
    if (isNum(r.val_pr_auc)) out += `<circle class="pt eo" cx="${x(i)}" cy="${y(r.val_pr_auc)}" r="3"><title>epoch ${i + 1} (${r.stage}) PR-AUC ${r.val_pr_auc.toFixed(4)} · loss ${f(r.train_loss, 4)} · ${r.sec}s</title></circle>`;
  });
  let bi = 0; pr.forEach((v, i) => { if (isNum(v) && v > pr[bi]) bi = i; });
  const anchor = x(bi) > W - m.r - 50 ? "end" : x(bi) < m.l + 50 ? "start" : "middle";
  const by = y(pr[bi]) - 13 < m.t + 4 ? y(pr[bi]) + 22 : y(pr[bi]) - 13;
  out += `<circle class="best-ring" cx="${x(bi)}" cy="${y(pr[bi])}" r="8"/><text x="${x(bi) + (anchor === "end" ? -10 : anchor === "start" ? 10 : 0)}" y="${by}" text-anchor="${anchor}" style="fill:var(--ink);font-size:11px">best ${pr[bi].toFixed(3)}</text>`;
  svg.innerHTML = out;
}

function runColor(name, names) { return RUN_COLORS[names.indexOf(name) % RUN_COLORS.length]; }

function chartAll(svg, runs, s) {
  const W = 640, H = 260, m = { l: 48, r: 16, t: 14, b: 28 };
  const series = runs.filter((r) => r.curve && r.curve.length).map((r) => ({ name: r.name, label: r.label, vals: r.curve, done: true }));
  if (s.current && s.epochs && s.epochs.length && !series.some((x) => x.name === s.current))
    series.push({ name: s.current, label: s.focus_label + " (live)", vals: s.epochs.map((e) => e.val_pr_auc), done: false });
  if (!series.length) { svg.innerHTML = `<text class="empty" x="${W / 2}" y="${H / 2}" text-anchor="middle">No finished runs yet</text>`; $("all-legend").innerHTML = ""; return; }
  const vals = series.flatMap((x) => x.vals).filter(isNum);
  let lo = Math.max(0.6, Math.min(...vals) - 0.02), hi = Math.min(1, Math.max(...vals) + 0.01);
  const n = Math.max(...series.map((x) => x.vals.length));
  const { x, y, g } = axes(W, H, m, lo, hi, n);
  const names = series.map((x) => x.name);
  let out = g;
  if (lo < 0.667) out += `<line class="ref-line" x1="${m.l}" x2="${W - m.r}" y1="${y(0.667)}" y2="${y(0.667)}"/><text class="ref-lbl" x="${W - m.r}" y="${y(0.667) - 4}" text-anchor="end">chance (prevalence)</text>`;
  series.forEach((sr) => {
    const c = runColor(sr.name, names);
    out += `<path class="ln" d="${pathOf(sr.vals, x, y)}" stroke="${c}" stroke-width="${sr.done ? 1.8 : 3}" ${sr.done ? 'opacity=".9"' : ""}><title>${esc(sr.label)}</title></path>`;
    const lastI = sr.vals.length - 1;
    if (isNum(sr.vals[lastI])) out += `<circle cx="${x(lastI)}" cy="${y(sr.vals[lastI])}" r="${sr.done ? 2.8 : 4}" fill="${c}"/>`;
  });
  svg.innerHTML = out;
  $("all-legend").innerHTML = series.map((sr) => `<span class="item"><i class="sw" style="background:${runColor(sr.name, names)}"></i>${esc(sr.label)}</span>`).join("");
}

/* ================= leaderboard (forest plot) ================= */
function renderForest(runs) {
  const svg = $("forest");
  const rows = runs.filter((r) => isNum(r[metric]));
  const lowerBetter = metric === "ece";
  rows.sort((a, b) => (lowerBetter ? a[metric] - b[metric] : b[metric] - a[metric]));
  const rowH = 34, top = 26, W = 640, L = 230, R = 70;
  const H = Math.max(120, top + rows.length * rowH + 30);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!rows.length) { svg.innerHTML = `<text class="empty" x="${W / 2}" y="60" text-anchor="middle">No finished runs yet</text>`; return; }
  const all = rows.flatMap((r) => [r[metric], ...(r.ci?.[metric] || [])]).filter(isNum);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 0.02; lo = Math.max(0, lo - pad); hi = Math.min(1, hi + pad);
  const x = (v) => L + ((v - lo) / (hi - lo || 1)) * (W - L - R);
  let out = `<g class="ax">`;
  for (let k = 0; k <= 4; k++) { const v = lo + ((hi - lo) * k) / 4; out += `<line x1="${x(v)}" x2="${x(v)}" y1="${top - 8}" y2="${H - 22}"/><text x="${x(v)}" y="${H - 6}" text-anchor="middle">${v.toFixed(2)}</text>`; }
  out += `</g>`;
  const base = rows.find((r) => r.name === "baseline_a_lr");
  if (base) out += `<line class="ref-line" x1="${x(base[metric])}" x2="${x(base[metric])}" y1="${top - 8}" y2="${H - 22}"/><text class="ref-lbl" x="${x(base[metric]) + 4}" y="${top - 12}">frozen-feature baseline</text>`;
  const names = runs.map((r) => r.name);
  rows.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2;
    const c = runColor(r.name, names);
    const ci = r.ci?.[metric];
    out += `<g class="f-row"><rect class="f-band" x="0" y="${cy - rowH / 2}" width="${W}" height="${rowH}"/>`;
    out += `<text class="f-name" x="8" y="${cy - 1}">${esc(r.label)}</text><text class="f-sub" x="8" y="${cy + 12}">${esc(r.name)}${r.epochs ? ` · ${r.epochs} ep` : ""}${r.vram_mb ? ` · ${r.vram_mb} MB` : ""}</text>`;
    if (ci) out += `<line class="f-ci" x1="${x(ci[0])}" x2="${x(ci[1])}" y1="${cy}" y2="${cy}"/><line class="f-cap" x1="${x(ci[0])}" x2="${x(ci[0])}" y1="${cy - 5}" y2="${cy + 5}"/><line class="f-cap" x1="${x(ci[1])}" x2="${x(ci[1])}" y1="${cy - 5}" y2="${cy + 5}"/>`;
    out += `<circle class="f-dot" cx="${x(r[metric])}" cy="${cy}" r="7" fill="${c}"><title>${esc(r.label)}: ${r[metric].toFixed(4)}${ci ? ` [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : " (CI after src.report)"}</title></circle>`;
    out += `<text class="f-val" x="${W - 8}" y="${cy + 4}" text-anchor="end">${r[metric].toFixed(3)}</text></g>`;
  });
  svg.innerHTML = out;
  $("forest-hint").textContent = `${lowerBetter ? "Lower is better. " : ""}Dot = point estimate at threshold 0.5. Whisker = 95% CI from resampling validation subjects 1000 times (shown once python -m src.report has run for that model). Overlapping whiskers mean the difference is not established.`;
}
$("metric-tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  metric = b.dataset.m;
  [...$("metric-tabs").children].forEach((x) => x.setAttribute("aria-selected", x === b ? "true" : "false"));
  if (lastState) renderForest(lastState.runs || []);
});

/* ================= data ================= */
function renderData(d) {
  if (!d) return;
  $("d-images").textContent = d.images.toLocaleString();
  $("d-subjects").textContent = d.subjects;
  const max = Math.max(...Object.values(d.splits).map((s) => s.images));
  $("d-splits").innerHTML = Object.entries(d.splits).map(([k, s]) => `
    <div class="split-row"><span class="nm">${esc(k)}</span>
      <div class="stack" style="width:${(100 * s.images) / max}%" title="${s.all} leukemic, ${s.hem} normal">
        <div class="a" style="flex:${s.all}"></div><div class="h" style="flex:${s.hem}"></div></div>
      <span class="num">${s.images.toLocaleString()} · ${s.subjects} subj</span></div>`).join("");
  const a = d.audit || {};
  $("d-audit").innerHTML = `
    <li><b>${a.corrupt ?? "–"}</b><span>all-zero (corrupt) files removed from the HF mirror</span></li>
    <li><b>${a.duplicates ?? "–"}</b><span>pixel-identical copies removed</span></li>
    <li><b>${(a.merged_groups || []).length}</b><span>patient-ID groups merged as one person: ${(a.merged_groups || []).map((g) => g.split(" ").join("=")).join(", ") || "none"}</span></li>
    <li><b>${d.patient_ids}</b><span>raw patient IDs → ${d.subjects} independent subjects; no subject spans two splits</span></li>`;
}

/* ================= tables / console ================= */
function renderEpochs(ep) {
  let bi = -1; ep.forEach((r, i) => { if (isNum(r.val_pr_auc) && (bi < 0 || r.val_pr_auc > ep[bi].val_pr_auc)) bi = i; });
  const vals = ep.map((r) => r.val_pr_auc).filter(isNum), lo = Math.min(...vals), hi = Math.max(...vals);
  $("epochs").tBodies[0].innerHTML = ep.map((r, i) => {
    const w = isNum(r.val_pr_auc) && hi > lo ? 6 + 44 * (r.val_pr_auc - lo) / (hi - lo) : 6;
    return `<tr class="${i === bi ? "best" : ""}"><td>${esc(r.stage)}</td><td>${i + 1}</td><td>${f(r.train_loss, 4)}</td>
      <td><span class="bar" style="width:${w}px"></span>${f(r.val_pr_auc, 4)}</td><td>${f(r.val_roc_auc, 4)}</td>
      <td>${f(r.val_recall, 3)}</td><td>${f(r.val_ece, 4)}</td><td>${r.sec}s</td></tr>`;
  }).reverse().join("") || `<tr><td colspan="8">No epochs yet</td></tr>`;
}
function renderConsole(s) {
  $("log-name").textContent = s.focus ? `${s.focus}.log` : "–";
  const pre = $("tail");
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
  pre.textContent = (s.tail || []).join("\n") || "no output yet";
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

/* ================= loop ================= */
async function tick() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    const s = await res.json();
    if (s.error) throw new Error(s.error);
    lastState = s;
    renderHeader(s);
    renderNow(s);
    renderGpu(s.gpu, s.detail);
    renderTimeline(s);
    chartRun($("ch-run"), s.epochs || []);
    chartAll($("ch-all"), s.runs || [], s);
    renderForest(s.runs || []);
    renderData(s.dataset);
    renderEpochs(s.epochs || []);
    renderConsole(s);
  } catch (e) {
    $("pulse").className = "pulse fail";
    $("state-text").textContent = "Monitor offline";
    $("queue-eta").textContent = String(e.message || e).slice(0, 80);
  } finally {
    setTimeout(tick, POLL_MS);
  }
}
tick();
