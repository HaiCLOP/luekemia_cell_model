export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const MAX_BYTES = 5 * 1024 * 1024;
export const ACCEPTED = ["image/png", "image/jpeg", "image/bmp", "image/x-ms-bmp"];

export type Decision = "leukemic" | "normal" | "refer_to_expert";

export interface PredictResponse {
  prediction_id: string;
  model_version: string;
  prob_raw: number;
  prob_calibrated: number;
  decision: Decision;
  band: { t_low: number; t_high: number };
  heatmap_png_base64: string | null;
  latency_ms: number;
  image_stored: boolean;
  disclaimer: string;
}

export interface StoredResult {
  response: PredictResponse;
  previewDataUrl: string;
  fileName: string;
  createdAt: string;
}

export const RESULT_KEY = "cell-triage:last-result";

async function errorText(res: Response): Promise<string> {
  try {
    const j = await res.json();
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return j.detail.map((d: { msg?: string }) => d.msg).join("; ");
  } catch {
    /* fall through */
  }
  return `The analysis service returned ${res.status}.`;
}

export async function predict(file: File, storeImage: boolean): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file);
  body.append("store_image", storeImage ? "true" : "false");
  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, { method: "POST", body });
  } catch {
    throw new Error("Could not reach the analysis service. Check that the API is running and try again.");
  }
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function sendFeedback(predictionId: string, userLabel: "leukemic" | "normal" | "unsure", comment: string) {
  const res = await fetch(`${API_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prediction_id: predictionId, user_label: userLabel, comment: comment.trim() || null }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

/** Downscale the uploaded image for display on the result page (keeps sessionStorage small). */
export function previewDataUrl(file: File, size = 448): Promise<string> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, size / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      c.getContext("2d")!.drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      resolve(c.toDataURL("image/png"));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("This file could not be read as an image."));
    };
    img.src = url;
  });
}
