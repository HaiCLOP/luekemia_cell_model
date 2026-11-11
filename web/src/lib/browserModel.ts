// Runs the deployed model (MH51401122, 5-network ONNX ensemble) inside the visitor's browser with ONNX Runtime Web.
// The image never leaves the device. This mirrors api/inference.py and api/quality.py step for step:
// input checks -> resize to 224 -> ImageNet normalisation -> five member logits -> mean probability ->
// temperature scaling -> refer-to-expert band, plus the out-of-distribution guard and the activation heatmap.
import meta from "@/data/model_meta.json";
import type { PredictResponse } from "@/lib/api";

export const MODEL_URL =
  process.env.NEXT_PUBLIC_MODEL_URL || "https://huggingface.co/Velion/MH51401122/resolve/main/MH51401122.onnx";
export const MODEL_MB = 76;
const CACHE_NAME = `cell-triage-model-${meta.onnx_sha256.slice(0, 12)}`;

type Ort = typeof import("onnxruntime-web");
let ortPromise: Promise<Ort> | null = null;
let sessionPromise: Promise<import("onnxruntime-web").InferenceSession> | null = null;

async function loadOrt(): Promise<Ort> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/wasm").then((ort) => {
      ort.env.wasm.wasmPaths = "/ort/";            // served by this site (scripts/copy-ort.mjs)
      ort.env.wasm.numThreads = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated ? 4 : 1;
      return ort as unknown as Ort;
    });
  }
  return ortPromise;
}

/** Downloads the model once (about 76 MB) and keeps it in the browser's cache storage for later visits. */
async function fetchModel(onProgress?: (fraction: number) => void): Promise<ArrayBuffer> {
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(MODEL_URL);
    if (hit) {
      onProgress?.(1);
      return hit.arrayBuffer();
    }
  } catch {
    cache = null;                                     // private windows may refuse cache storage; just download
  }
  const res = await fetch(MODEL_URL);
  if (!res.ok || !res.body) throw new Error("Could not download the model. Check your connection and try again.");
  const total = Number(res.headers.get("content-length")) || MODEL_MB * 1024 * 1024;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(Math.min(0.99, got / total));
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  // refuse a file that is not the deployed model
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", buf)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  if (digest !== meta.onnx_sha256) throw new Error("The downloaded model does not match the expected version.");
  try {
    await cache?.put(MODEL_URL, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
  } catch {
    /* storage full or blocked: the model still runs this time */
  }
  onProgress?.(1);
  return buf.buffer;
}

export async function getSession(onProgress?: (fraction: number) => void) {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await loadOrt();
      const model = await fetchModel(onProgress);
      return ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    })();
    sessionPromise.catch(() => {
      sessionPromise = null;                          // allow a retry after a failed download
    });
  } else {
    onProgress?.(1);
  }
  return sessionPromise;
}

// ------------------------------------------------------------------ input checks (api/quality.py)
const THUMB = 64, DARK = 24, MAX_ASPECT = 1.25, MIN_BORDER_DARK = 0.7;
const MIN_FG = 0.02, MAX_FG = 0.6, MIN_MAIN_BLOB = 0.8, MIN_CONTRAST = 1.0;

export class InputRejected extends Error {
  constructor(public code: string, message: string) {
    super(message);
  }
}

/** A decoded image: its size and something a canvas can draw. */
export interface Decoded {
  width: number;
  height: number;
  el: CanvasImageSource;
}

const MAX_SIDE = 2048; // phone photos can be 4000+ px; canvases on some phones refuse very large images

/**
 * Decode an uploaded file. First choice: no colour-space conversion and no alpha premultiplication, so pixels match
 * what Python reads. Some mobile browsers reject those options ("the source image could not be decoded"), so fall
 * back to a plain decode, then to an <img> element. Very large photos are scaled down to at most 2048 px first.
 */
export async function decodeImage(file: Blob): Promise<Decoded> {
  let src: Decoded | null = null;
  for (const opts of [{ colorSpaceConversion: "none", premultiplyAlpha: "none" } as ImageBitmapOptions, undefined]) {
    try {
      const bmp = await createImageBitmap(file, opts);
      src = { width: bmp.width, height: bmp.height, el: bmp };
      break;
    } catch {
      /* try the next way */
    }
  }
  if (!src) {
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.decoding = "async";
      img.src = url;
      await img.decode();
      src = { width: img.naturalWidth, height: img.naturalHeight, el: img };
    } catch {
      throw new InputRejected("unsupported_input", "This browser could not read the image. Try saving it as PNG or JPEG and upload again.");
    } finally {
      setTimeout(() => URL.revokeObjectURL(url), 0);
    }
  }
  const side = Math.max(src.width, src.height);
  if (side > MAX_SIDE) {
    const k = MAX_SIDE / side;
    const c = document.createElement("canvas");
    c.width = Math.round(src.width * k);
    c.height = Math.round(src.height * k);
    const ctx = c.getContext("2d")!;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(src.el, 0, 0, c.width, c.height);
    src = { width: c.width, height: c.height, el: c };
  }
  return src;
}

/** The image's own pixels at native size, RGBA, with no resampling. */
function nativePixels(img: Decoded): Uint8ClampedArray {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img.el, 0, 0);
  return ctx.getImageData(0, 0, img.width, img.height).data;
}

// Pillow's BILINEAR resize, reproduced exactly (src/libImaging/Resample.c): a triangle filter whose support grows
// with the shrink factor (antialiasing), 22-bit fixed-point weights, horizontal pass first, rounding to 8 bits in
// between. Browsers' own canvas scaling differs from this and from each other, which moved scores by up to 0.02.
const PRECISION_BITS = 32 - 8 - 2;

function coeffs(inSize: number, outSize: number) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = 1.0 * filterScale;
  const bounds: number[] = [], weights: Int32Array[] = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterScale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k: number[] = [];
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      const w = t < 1 ? 1 - t : 0;
      k.push(w);
      ww += w;
    }
    const ki = new Int32Array(xmax);
    for (let x = 0; x < xmax; x++) {
      const v = ww ? k[x] / ww : 0;
      ki[x] = Math.trunc(v * (1 << PRECISION_BITS) + (v < 0 ? -0.5 : 0.5));
    }
    bounds.push(xmin);
    weights.push(ki);
  }
  return { bounds, weights };
}

const clip8 = (v: number) => (v <= 0 ? 0 : v >= 255 ? 255 : v);

/** RGBA in -> RGB out, resized like PIL Image.resize((w, h), Image.BILINEAR). */
function resizePIL(src: Uint8ClampedArray, inW: number, inH: number, outW: number, outH: number): Uint8Array {
  const half = 1 << (PRECISION_BITS - 1);
  // horizontal pass: inH rows x outW columns
  const hc = coeffs(inW, outW);
  const tmp = new Uint8Array(inH * outW * 3);
  for (let y = 0; y < inH; y++)
    for (let xx = 0; xx < outW; xx++) {
      const xmin = hc.bounds[xx], k = hc.weights[xx];
      for (let ch = 0; ch < 3; ch++) {
        let acc = half;
        for (let x = 0; x < k.length; x++) acc += src[4 * (y * inW + xmin + x) + ch] * k[x];
        tmp[3 * (y * outW + xx) + ch] = clip8(Math.floor(acc / (1 << PRECISION_BITS)));
      }
    }
  // vertical pass
  const vc = coeffs(inH, outH);
  const out = new Uint8Array(outH * outW * 3);
  for (let yy = 0; yy < outH; yy++) {
    const ymin = vc.bounds[yy], k = vc.weights[yy];
    for (let xx = 0; xx < outW; xx++)
      for (let ch = 0; ch < 3; ch++) {
        let acc = half;
        for (let y = 0; y < k.length; y++) acc += tmp[3 * ((ymin + y) * outW + xx) + ch] * k[y];
        out[3 * (yy * outW + xx) + ch] = clip8(Math.floor(acc / (1 << PRECISION_BITS)));
      }
  }
  return out;
}

function pixels(img: Decoded, w: number, h: number): Uint8Array {
  return resizePIL(nativePixels(img), img.width, img.height, w, h);
}

function largestBlobShare(fg: Uint8Array, n: number): number {
  const total = fg.reduce((a, b) => a + b, 0);
  if (!total) return 0;
  const seen = new Uint8Array(n * n);
  let best = 0;
  const stack: number[] = [];
  for (let s = 0; s < n * n; s++) {
    if (!fg[s] || seen[s]) continue;
    let size = 0;
    seen[s] = 1;
    stack.push(s);
    while (stack.length) {
      const i = stack.pop()!;
      size++;
      const y = Math.floor(i / n), x = i % n;
      for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || yy >= n || xx < 0 || xx >= n) continue;
        const j = yy * n + xx;
        if (fg[j] && !seen[j]) {
          seen[j] = 1;
          stack.push(j);
        }
      }
    }
    best = Math.max(best, size);
  }
  return best / total;
}

export function checkInput(img: Decoded): void {
  const w = img.width, h = img.height;
  const px = pixels(img, THUMB, THUMB);
  const n = THUMB;
  const fg = new Uint8Array(n * n);
  const grey = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    const r = px[3 * i], g = px[3 * i + 1], b = px[3 * i + 2];
    fg[i] = Math.max(r, g, b) >= DARK ? 1 : 0;
    grey[i] = (r + g + b) / 3;
  }
  let borderDark = 0, borderN = 0, fgN = 0;
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const i = y * n + x;
      fgN += fg[i];
      if (y < 3 || y >= n - 3 || x < 3 || x >= n - 3) {
        borderN++;
        borderDark += 1 - fg[i];
      }
    }
  let inner = fg.slice();
  for (let k = 0; k < 2; k++) {                       // two-pixel erosion: the interior of the cell only
    const next = new Uint8Array(n * n);
    for (let y = 1; y < n - 1; y++)
      for (let x = 1; x < n - 1; x++) {
        const i = y * n + x;
        next[i] = inner[i] & inner[i - n] & inner[i + n] & inner[i - 1] & inner[i + 1];
      }
    inner = next;
  }
  let m = 0, c = 0;
  for (let i = 0; i < n * n; i++) if (inner[i]) { m += grey[i]; c++; }
  m = c ? m / c : 0;
  let v = 0;
  for (let i = 0; i < n * n; i++) if (inner[i]) v += (grey[i] - m) ** 2;
  const contrast = c ? Math.sqrt(v / c) : 0;
  const fgShare = fgN / (n * n);

  if (Math.max(w, h) / Math.max(1, Math.min(w, h)) > MAX_ASPECT)
    throw new InputRejected("unsupported_input", "The image is not close to square. Upload one cropped cell in a square frame.");
  if (fgShare < MIN_FG) throw new InputRejected("insufficient_image_quality", "The image is almost entirely black. No cell was found.");
  if (borderDark / borderN < MIN_BORDER_DARK)
    throw new InputRejected("unsupported_input", "This looks like a smear photo or an uncropped field, not one cell cut out on a black background. This page reads one segmented cell; whole smears need the Smear page (local API for now).");
  if (fgShare > MAX_FG)
    throw new InputRejected("unsupported_input", "The cell fills the whole frame. The model expects a cell with black background around it.");
  if (largestBlobShare(fg, n) < MIN_MAIN_BLOB)
    throw new InputRejected("unsupported_input", "More than one object was found. Upload a single cell per image.");
  if (contrast < MIN_CONTRAST)
    throw new InputRejected("insufficient_image_quality", "The cell has almost no internal detail (flat or badly out of focus).");
}

// ------------------------------------------------------------------ heatmap (api/inference.py)
const LUT_STOPS = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
const LUT: number[][] = Array.from({ length: 256 }, (_, i) => {
  const t = (i / 255) * (LUT_STOPS.length - 1);
  const k = Math.min(LUT_STOPS.length - 2, Math.floor(t)), f = t - k;
  return [0, 1, 2].map((ch) => LUT_STOPS[k][ch] * (1 - f) + LUT_STOPS[k + 1][ch] * f);
});

function heatmap(cam: Float32Array, side: number, rgb: Uint8Array, size: number): string {
  let lo = Infinity, hi = -Infinity;
  for (const v of cam) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const small = document.createElement("canvas");
  small.width = small.height = side;
  const sctx = small.getContext("2d")!;
  const sd = sctx.createImageData(side, side);
  for (let i = 0; i < side * side; i++) {
    const g = Math.round(((cam[i] - lo) / (hi - lo + 1e-8)) * 255);
    sd.data.set([g, g, g, 255], 4 * i);
  }
  sctx.putImageData(sd, 0, 0);
  const big = document.createElement("canvas");
  big.width = big.height = size;
  const bctx = big.getContext("2d")!;
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = "high";
  bctx.drawImage(small, 0, 0, size, size);
  const up = bctx.getImageData(0, 0, size, size);
  for (let i = 0; i < size * size; i++) {
    const c = LUT[up.data[4 * i]];
    for (let ch = 0; ch < 3; ch++) up.data[4 * i + ch] = Math.round(0.55 * rgb[3 * i + ch] + 0.45 * c[ch]);
    up.data[4 * i + 3] = 255;
  }
  bctx.putImageData(up, 0, 0);
  return big.toDataURL("image/png").split(",")[1];
}

// ------------------------------------------------------------------ prediction
const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));

export async function predictInBrowser(file: File, onProgress?: (fraction: number) => void): Promise<PredictResponse> {
  const img = await decodeImage(file);
  if (Math.min(img.width, img.height) < 32)
    throw new InputRejected("insufficient_image_quality", "Image is too small (minimum 32 x 32 px).");
  checkInput(img);
  const session = await getSession(onProgress);
  const ort = await loadOrt();
  const t0 = performance.now();

  const size = meta.input.size;
  const rgb = pixels(img, size, size);
  const x = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++)
    for (let ch = 0; ch < 3; ch++)
      x[ch * size * size + i] = (rgb[3 * i + ch] / 255 - meta.input.mean[ch]) / meta.input.std[ch];
  const out = await session.run({ image: new ort.Tensor("float32", x, [1, 3, size, size]) });
  const logits = out.logits.data as Float32Array;
  const cam = out.cam.data as Float32Array;

  const maxAbs = Math.max(...Array.from(logits, (v) => Math.abs(v)));
  if (maxAbs > (meta.ood_max_abs_logit ?? 25))
    throw new InputRejected("out_of_distribution",
      "The model's internal scores for this image are far outside anything it produced on its training data, so any answer would be meaningless. Usually the stain colours, magnification or image source differ from C-NMC.");

  const members = Array.from(logits, sigmoid);
  const pRaw = Math.min(1 - 1e-6, Math.max(1e-6, members.reduce((a, b) => a + b, 0) / members.length));
  const pCal = sigmoid(Math.log(pRaw / (1 - pRaw)) / meta.temperature);
  const decision = pCal < meta.t_low ? "normal" : pCal > meta.t_high ? "leukemic" : "refer_to_expert";
  const side = Math.round(Math.sqrt(cam.length));
  const heat = heatmap(cam, side, rgb, size);
  return {
    prediction_id: crypto.randomUUID(),
    model_version: meta.name,
    prob_raw: Math.round(pRaw * 1e4) / 1e4,
    prob_calibrated: Math.round(pCal * 1e4) / 1e4,
    decision,
    band: { t_low: meta.t_low, t_high: meta.t_high },
    heatmap_png_base64: heat,
    latency_ms: Math.round(performance.now() - t0),
    image_stored: false,
    disclaimer: "Research demo. Not for clinical use.",
  };
}
