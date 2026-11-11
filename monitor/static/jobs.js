// Plain-language progress for long jobs. Each job may describe itself (title, explain, steps); for jobs
// that do not, the FRIENDLY table below supplies the words. Technical details stay behind a toggle.
const $ = (id) => document.getElementById(id);

const FRIENDLY = [
  {
    match: /^taleqani_eval/,
    title: "Testing the smear reader on another hospital's blood smears",
    explain: "The model was trained on cells from one lab. This checks how well it does on about 3,200 smear photos " +
      "from a different hospital: find every white cell, cut it out, and ask the model about it. " +
      "The result tells us whether it works outside the lab it learned from.",
    steps: [
      { key: /loading/, text: "Open the 3,200 smear photos", weight: 3 },
      { key: /red cells/, text: "Measure the red cells in every photo, to work out the zoom level", weight: 4 },
      { key: /finding and scoring/, text: "Find each white cell and ask the model about it", weight: 90 },
      { key: /metrics/, text: "Add up the results and work out how reliable they are", weight: 3 },
    ],
  },
  {
    match: /^bccd/,
    title: "Checking how well white cells are found",
    explain: "Runs the cell finder on 364 labelled smear photos from healthy donors and compares what it finds with where people marked the cells.",
    steps: [
      { key: /matching/, text: "Find white cells and compare with the hand-drawn boxes", weight: 40 },
      { key: /healthy/, text: "Ask the model about the normal cells from healthy donors", weight: 60 },
    ],
  },
];

function friendly(job) {
  const f = FRIENDLY.find((x) => x.match.test(job.job)) || {};
  return {
    title: f.title || job.plain_title || job.title || job.job,
    explain: f.explain || job.explain || "",
    steps: f.steps || (job.steps ? job.steps.map((t) => ({ key: null, text: t, weight: 1 })) : []),
  };
}

function minutes(s) {
  if (s == null || !isFinite(s)) return null;
  const m = Math.round(s / 60);
  if (s < 60) return "less than a minute";
  if (m < 60) return `about ${m} minute${m === 1 ? "" : "s"}`;
  const h = Math.floor(m / 60), r = m % 60;
  return `about ${h} hour${h === 1 ? "" : "s"}${r ? ` ${r} min` : ""}`;
}
const clock = (d) => d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

function stepIndex(job, steps) {
  if (job.step_index != null) return job.step_index;
  const i = steps.findIndex((s) => s.key && s.key.test(job.stage || ""));
  return i;
}

function render(jobs) {
  const box = $("jobs");
  $("empty").hidden = jobs.length > 0;
  box.replaceChildren();
  for (const j of jobs) {
    const f = friendly(j);
    const el = $("job-tpl").content.firstElementChild.cloneNode(true);
    el.dataset.status = j.status;
    el.querySelector(".title").textContent = f.title;
    el.querySelector(".explain").textContent = f.explain;
    el.querySelector(".badge").textContent =
      { running: "Working", finished: "Done", failed: "Stopped with an error", stale: "Not responding" }[j.status] || j.status;

    // overall progress: earlier steps count as complete, the current one by its own fraction; steps are
    // weighted by how long they usually take, so a quick first step does not make the bar jump ahead
    const idx = stepIndex(j, f.steps);
    const within = Math.min(1, j.done / Math.max(1, j.total));
    let frac = within;
    if (f.steps.length && idx >= 0) {
      const w = f.steps.map((s) => s.weight || 1);
      const total = w.reduce((a, b) => a + b, 0);
      frac = (w.slice(0, idx).reduce((a, b) => a + b, 0) + within * w[idx]) / total;
    }
    if (j.status === "finished") frac = 1;
    el.querySelector(".pct").textContent = `${Math.floor(100 * frac)}%`;
    el.querySelector(".fill").style.width = `${(100 * frac).toFixed(1)}%`;
    el.querySelector(".bar").setAttribute("aria-valuenow", Math.floor(100 * frac));

    const t = el.querySelector(".time");
    if (j.status === "running") {
      const left = minutes(j.eta_s);
      t.textContent = left ? `${left} left · done around ${clock(new Date(Date.now() + j.eta_s * 1000))}` : "working out how long it will take…";
    } else if (j.status === "finished") {
      t.textContent = `finished at ${clock(new Date(j.updated))}`;
    } else if (j.status === "stale") {
      t.textContent = "no update for over 2 minutes; it may have been stopped";
    } else {
      t.textContent = "";
    }

    const now = el.querySelector(".now");
    if (j.status === "running" && j.total > 1) {
      now.textContent = `This step: ${j.done.toLocaleString()} of ${j.total.toLocaleString()} done. Running for ${minutes((Date.now() - Date.parse(j.started)) / 1000).replace("about ", "")}.`;
    } else if (j.status === "failed") {
      now.textContent = `Error: ${j.note}`;
    }

    const ol = el.querySelector(".steps");
    f.steps.forEach((s, i) => {
      const li = document.createElement("li");
      li.textContent = s.text;
      if (j.status === "finished" || i < idx) li.className = "done";
      else if (i === idx) li.className = j.status === "running" ? "current" : "";
      ol.append(li);
    });
    if (!f.steps.length) ol.remove();

    const tech = el.querySelector(".tech");
    const rows = { "job id": j.job, "stage": j.stage, "items": `${j.done} / ${j.total}`, "speed": j.rate_per_s ? `${j.rate_per_s.toFixed(2)} per second` : "–", "last item": j.note || "–", ...j.metrics };
    for (const [k, v] of Object.entries(rows)) {
      const dt = document.createElement("dt"); dt.textContent = k.replace(/_/g, " ");
      const dd = document.createElement("dd"); dd.textContent = typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3)) : String(v);
      tech.append(dt, dd);
    }
    box.append(el);
  }
}

async function tick() {
  try {
    const data = await (await fetch("/api/jobs", { cache: "no-store" })).json();
    const jobs = (data.jobs || []).filter((j) => !/_n\d+$/.test(j.job) || j.status === "running"); // hide finished test runs
    render(jobs);
    const running = jobs.filter((j) => j.status === "running").length;
    $("state").textContent = running ? `${running === 1 ? "One job" : running + " jobs"} running. You can leave this page open.` : "Nothing running right now.";
  } catch {
    $("state").textContent = "Can't reach the monitor. Is monitor/server.py still running?";
  }
}

setInterval(tick, 2000);
tick();
