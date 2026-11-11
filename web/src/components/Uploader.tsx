"use client";

import { useRouter } from "next/navigation";
import { useId, useRef, useState } from "react";
import { ACCEPTED, MAX_BYTES, RESULT_KEY, predict, previewDataUrl, type StoredResult } from "@/lib/api";

const SAMPLES = [
  { file: "/samples/sample-leukemic.png", label: "Leukemic cell" },
  { file: "/samples/sample-normal.png", label: "Normal cell" },
  { file: "/samples/sample-borderline.png", label: "Borderline cell" },
];

export function Uploader() {
  const router = useRouter();
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [store, setStore] = useState(false);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function choose(f: File | undefined | null) {
    setError(null);
    if (!f) return;
    const okType = ACCEPTED.includes(f.type) || /\.(png|jpe?g|bmp)$/i.test(f.name);
    if (!okType) return setError("Please choose a PNG, JPEG or BMP image.");
    if (f.size > MAX_BYTES) return setError("That image is larger than 5 MB. Please choose a smaller one.");
    try {
      setPreview(await previewDataUrl(f));
      setFile(f);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function run(f: File) {
    setBusy(true);
    setError(null);
    try {
      const [response, dataUrl] = await Promise.all([predict(f, store), preview ?? previewDataUrl(f)]);
      const stored: StoredResult = { response, previewDataUrl: dataUrl, fileName: f.name, createdAt: new Date().toISOString() };
      sessionStorage.setItem(RESULT_KEY, JSON.stringify(stored));
      router.push("/result");
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  async function trySample(path: string, label: string) {
    setError(null);
    const blob = await (await fetch(path)).blob();
    const f = new File([blob], `${label.toLowerCase().replace(/\s+/g, "-")}.png`, { type: "image/png" });
    setFile(f);
    const url = await previewDataUrl(f);
    setPreview(url);
    await run(f);
  }

  return (
    <section className="tray" aria-labelledby="upload-title" id="analyse">
      <div className="tray-head">
        <h2 id="upload-title" className="eyebrow">Specimen tray · analyse one cell</h2>
        <span className="label">PNG · JPEG · BMP · up to 5 MB</span>
      </div>

      <label
        htmlFor={inputId}
        className={`drop${drag ? " drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files?.[0]); }}
      >
        <span className="well" aria-hidden="true">
          <svg width="30" height="30" viewBox="0 0 30 30"><circle cx="15" cy="15" r="11" fill="none" stroke="#1d2340" strokeWidth="1.5" /><circle cx="16.5" cy="14" r="6" fill="#2b3a78" /><circle cx="18" cy="12.5" r="1.6" fill="#d4573a" /></svg>
        </span>
        <span>
          <span className="big" style={{ display: "block" }}>Drop a cell image here</span>
          <span className="small">or press to choose a file. One segmented white blood cell per image works best.</span>
        </span>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,image/png,image/jpeg,image/bmp"
          onChange={(e) => choose(e.target.files?.[0])}
        />
      </label>

      {preview && file && (
        <div className="preview">
          <img src={preview} alt="Preview of the selected cell image" />
          <div>
            <div style={{ fontWeight: 600 }}>{file.name}</div>
            <div className="small">{(file.size / 1024).toFixed(0)} KB, ready to analyse</div>
          </div>
        </div>
      )}

      <label className="check">
        <input type="checkbox" checked={store} onChange={(e) => setStore(e.target.checked)} />
        <span>Keep my image for 24 hours to help improve the model. Off by default: otherwise only a fingerprint (SHA-256) of the file is recorded.</span>
      </label>

      <div className="actions">
        <button type="button" className="btn btn-primary" disabled={!file || busy} onClick={() => file && run(file)}>
          {busy ? "Analysing..." : "Analyse cell"}
        </button>
        {file && !busy && (
          <button type="button" className="btn btn-outline" onClick={() => { setFile(null); setPreview(null); if (inputRef.current) inputRef.current.value = ""; }}>
            Clear
          </button>
        )}
        <span role="status" aria-live="polite" className="small">{busy ? "Running the five-model ensemble, usually under a second." : ""}</span>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      <div className="samples">
        <span className="small">No image to hand? Try a validation cell:</span>
        {SAMPLES.map((s) => (
          <button key={s.file} type="button" className="sample" disabled={busy} onClick={() => trySample(s.file, s.label)}>
            <img src={s.file} alt="" />
            {s.label}
          </button>
        ))}
      </div>
    </section>
  );
}
