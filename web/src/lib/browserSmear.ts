// Smear analysis entirely in the browser. Mirrors api/smear.py:
//   Python cell finder (src/vision/segment.py, run by Pyodide in a web worker) -> C-NMC-style crops
//   -> learned cell-type filter (ONNX) -> MH51401122 ensemble (ONNX) with the out-of-distribution guard
//   -> calibration and band -> smear summary and an annotated overlay.
import ctMeta from "@/data/cell_type_meta.json";
import meta from "@/data/model_meta.json";
import stain from "@/data/cnmc_stain.json";
import type { Decision, SmearCell, SmearResponse } from "@/lib/api";
import { getSession, loadOrt, sessionFrom } from "@/lib/browserModel";

const FILTER_URL =
  process.env.NEXT_PUBLIC_FILTER_URL || "https://huggingface.co/Velion/MH51401122/resolve/main/cell_type_filter.onnx";
const COLOURS: Record<string, string> = { leukemic: "#d4573a", normal: "#4f7a5a", refer_to_expert: "#a86b12", not_scored: "#7d8b93" };

interface WorkerOut {
  image: { data: Uint8Array; shape: number[] };
  crops: { data: Uint8Array; shape: number[] };
  filterCrops: { data: Uint8Array; shape: number[] };
  width: number;
  height: number;
  shrink: number;
  source: SmearResponse["scale_source"];
  rbc: number | null;
  scale: number | null;
  notes: string[];
  cells: { box: number[]; lymphoid_like: boolean; touches_border: boolean; contour: number[][] }[];
  scored: number[];
}

let worker: Worker | null = null;
let nextId = 0;

function runWorker(file: File, umPerPx: number | undefined, onStatus: (s: string) => void): Promise<WorkerOut> {
  if (!worker) worker = new Worker("/smear-worker.js");
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const w = worker!;
    const handler = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "status") return onStatus(m.text);
      if (m.id !== id) return;
      w.removeEventListener("message", handler);
      if (m.type === "result") resolve(m.out as WorkerOut);
      else reject(new Error(`The cell finder failed: ${m.message}`));
    };
    w.addEventListener("message", handler);
    w.postMessage({ id, file, umPerPx, stainJson: JSON.stringify(stain), modelSize: meta.input.size, filterSize: ctMeta.input_size });
  });
}

let filterPromise: ReturnType<typeof sessionFrom> | null = null;

function normalise(u8: Uint8Array, n: number, size: number, mean: number[], std: number[]): Float32Array {
  const x = new Float32Array(n * 3 * size * size);
  const plane = size * size;
  for (let b = 0; b < n; b++)
    for (let i = 0; i < plane; i++)
      for (let ch = 0; ch < 3; ch++)
        x[b * 3 * plane + ch * plane + i] = (u8[(b * plane + i) * 3 + ch] / 255 - mean[ch]) / std[ch];
  return x;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));

function decide(p: number): Decision {
  return p < meta.t_low ? "normal" : p > meta.t_high ? "leukemic" : "refer_to_expert";
}

function overlay(out: WorkerOut, decisions: string[]): string {
  const { width: w, height: h } = out;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    img.data[4 * i] = out.image.data[3 * i];
    img.data[4 * i + 1] = out.image.data[3 * i + 1];
    img.data[4 * i + 2] = out.image.data[3 * i + 2];
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const lw = Math.max(2, Math.round(Math.max(w, h) / 400));
  ctx.lineWidth = lw;
  ctx.font = `${10 + 4 * lw}px sans-serif`;
  out.cells.forEach((cell, i) => {
    ctx.strokeStyle = ctx.fillStyle = COLOURS[decisions[i]];
    if (cell.contour.length) {
      ctx.beginPath();
      cell.contour.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.stroke();
    }
    ctx.fillText(String(i + 1), cell.box[0], Math.max(12, cell.box[1] - 4));
  });
  return c.toDataURL("image/png").split(",")[1];
}

export async function analyzeSmearInBrowser(file: File, umPerPx: number | undefined, onStatus: (s: string) => void): Promise<SmearResponse> {
  const t0 = performance.now();
  const modelReady = getSession((f) => f < 1 && onStatus(`Downloading the cell model, ${Math.round(100 * f)}% (about 76 MB, only the first time)`));
  if (!filterPromise) {
    filterPromise = sessionFrom(FILTER_URL, ctMeta.onnx_sha256);
    filterPromise.catch(() => { filterPromise = null; });
  }
  const out = await runWorker(file, umPerPx, onStatus);
  onStatus("Scoring each cell with the five-model ensemble");
  const [session, filter, ort] = await Promise.all([modelReady, filterPromise, loadOrt()]);
  const warnings = [...out.notes];
  if (out.source === "unknown")
    warnings.push("Scale could not be measured; cells were resized to a typical C-NMC size, which removes size information.");
  else if (out.source === "red_cells_in_image")
    warnings.push("Scale was measured from the red cells in this one image (about 20% error per image on BCCD). If you know the pixel size, enter it for a steadier result.");

  const n = out.scored.length;
  const fs = ctMeta.input_size, ms = meta.input.size;
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  let keep: boolean[] = [];
  if (n) {
    const fz = (await filter.run({ image: new ort.Tensor("float32", normalise(out.filterCrops.data, n, fs, mean, std), [n, 3, fs, fs]) })).logit.data as Float32Array;
    keep = Array.from(fz, (z) => sigmoid(z) >= ctMeta.threshold);
  }
  const kept = out.scored.filter((_, k) => keep[k]);
  const probs = new Map<number, number>(), ood = new Set<number>(), dropped = new Set(out.scored.filter((_, k) => !keep[k]));
  if (kept.length) {
    const plane = ms * ms * 3;
    const sel = new Uint8Array(kept.length * plane);
    out.scored.forEach((cellIdx, k) => {
      const j = kept.indexOf(cellIdx);
      if (j >= 0) sel.set(out.crops.data.subarray(k * plane, (k + 1) * plane), j * plane);
    });
    const res = await session.run({ image: new ort.Tensor("float32", normalise(sel, kept.length, ms, meta.input.mean, meta.input.std), [kept.length, 3, ms, ms]) });
    const L = res.logits.data as Float32Array;
    const m = L.length / kept.length;
    kept.forEach((cellIdx, j) => {
      const row = Array.from(L.subarray(j * m, (j + 1) * m));
      if (Math.max(...row.map(Math.abs)) > (meta.ood_max_abs_logit ?? 25)) return void ood.add(cellIdx);
      const pRaw = Math.min(1 - 1e-6, Math.max(1e-6, row.map(sigmoid).reduce((a, b) => a + b, 0) / row.length));
      probs.set(cellIdx, sigmoid(Math.log(pRaw / (1 - pRaw)) / meta.temperature));
    });
  }

  const cells: SmearCell[] = [];
  const decisions: string[] = [];
  out.cells.forEach((c, i) => {
    let decision: SmearCell["decision"] = "not_scored", p: number | null = null, reason: string | null = null;
    if (probs.has(i)) {
      p = probs.get(i)!;
      decision = decide(p);
    } else {
      reason = ood.has(i) ? "out of distribution: the model's internal scores were far outside their normal range"
        : dropped.has(i) ? "not a lymphocyte according to the cell-type filter; the model only knows lymphocytes"
        : c.touches_border ? "touches the image edge"
        : "not lymphoid-like (lobed or irregular nucleus); the model only knows lymphocytes";
    }
    decisions.push(decision);
    cells.push({ id: i + 1, box: c.box.map((v) => Math.round(v * out.shrink)), lymphoid_like: c.lymphoid_like,
      touches_border: c.touches_border, prob_calibrated: p == null ? null : Math.round(p * 1e4) / 1e4, decision, reason });
  });
  const scored = [...probs.values()];
  const meanP = scored.length ? scored.reduce((a, b) => a + b, 0) / scored.length : null;
  if (scored.length && scored.length < 3) warnings.push(`Only ${scored.length} cell(s) could be scored; the smear summary is weak.`);
  const count = (d: string) => decisions.filter((x) => x === d).length;
  return {
    model_version: meta.name,
    rbc_diameter_px: out.rbc == null ? null : Math.round(out.rbc * out.shrink * 10) / 10,
    scale_source: out.source,
    scale_to_cnmc: out.scale == null ? null : Math.round((out.scale / out.shrink) * 1e4) / 1e4,
    band: { t_low: meta.t_low, t_high: meta.t_high },
    summary: {
      cells_found: out.cells.length, cells_scored: scored.length, leukemic: count("leukemic"), normal: count("normal"),
      refer_to_expert: count("refer_to_expert"), not_scored: count("not_scored"),
      mean_prob_calibrated: meanP == null ? null : Math.round(meanP * 1e4) / 1e4,
      smear_decision: meanP == null ? "no_scorable_cells" : decide(meanP),
    },
    cells,
    overlay_png_base64: overlay(out, decisions),
    warnings,
    latency_ms: Math.round(performance.now() - t0),
    disclaimer: "Research demo. Not for clinical use.",
  };
}
