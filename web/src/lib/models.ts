// The two models the site offers. MH51401102 is the default; MH51401122 is kept as a selectable second model.
import meta102 from "@/data/model_meta.json";
import meta122 from "@/data/MH51401122_meta.json";

export type ModelName = "MH51401102" | "MH51401122";
export const DEFAULT_MODEL: ModelName = "MH51401102";

export interface ModelInfo {
  name: ModelName;
  label: string;
  what: string;
  url: string;
  sha256: string;
  mb: number;
  input: { size: number; mean: number[]; std: number[] };
  temperature: number;
  t_low: number;
  t_high: number;
  ood_max_abs_logit: number;
}

export const MODELS: Record<ModelName, ModelInfo> = {
  MH51401102: {
    name: "MH51401102",
    label: "MH51401102 · three labs (default)",
    what: `Blood-cell foundation model (DinoBloom) with a classifier and a cell filter trained on three labs. ROC-AUC ${meta102.metrics.all_idb2.roc_auc.toFixed(2)} on ALL-IDB2, a lab it never saw.`,
    // Checked against model_meta.json's SHA-256 before use. (The same bytes are also on Hugging Face under their
    // pre-rename name, dinobloom_b_3lab.onnx, for browsers that cached that URL.)
    url: process.env.NEXT_PUBLIC_MODEL_URL || "https://huggingface.co/Velion/MH51401122/resolve/main/MH51401102.onnx",
    sha256: meta102.onnx_sha256,
    mb: Math.round(meta102.onnx_mb),
    input: meta102.input,
    temperature: meta102.temperature,
    t_low: meta102.t_low,
    t_high: meta102.t_high,
    ood_max_abs_logit: meta102.ood_max_abs_logit,
  },
  MH51401122: {
    name: "MH51401122",
    label: "MH51401122 · C-NMC only",
    what: `Five EfficientNet-B0 networks fine-tuned on C-NMC, with a heatmap. C-NMC test PR-AUC ${meta122.metrics.test.pr_auc.toFixed(2)}, but ROC-AUC ${meta102.metrics.all_idb2.retired_MH51401122_roc_auc.toFixed(2)} on ALL-IDB2: it does not transfer well to other labs.`,
    url: process.env.NEXT_PUBLIC_MODEL_MH51401122_URL || "https://huggingface.co/Velion/MH51401122/resolve/main/MH51401122.onnx",
    sha256: meta122.onnx_sha256,
    mb: 76,
    input: meta122.input,
    temperature: meta122.temperature,
    t_low: meta122.t_low,
    t_high: meta122.t_high,
    ood_max_abs_logit: meta122.ood_max_abs_logit,
  },
};

export const filterThresholdLogit = meta102.filter_threshold_logit;
