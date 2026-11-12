"use client";

import { useId, useRef, useState } from "react";
import { ACCEPTED, RUN_IN_BROWSER, SMEAR_MAX_BYTES, analyzeSmear, type SmearDecision, type SmearResponse } from "@/lib/api";

const SAMPLES = [
  { file: "/samples/smear-bccd-1.jpg", label: "Sample A: granulocytes" },
  { file: "/samples/smear-bccd-2.jpg", label: "Sample B: healthy lymphocyte" },
];

const LABEL: Record<SmearDecision | "no_scorable_cells", string> = {
  leukemic: "Leukemic",
  normal: "Normal",
  refer_to_expert: "Refer to expert",
  not_scored: "Not scored",
  no_scorable_cells: "No scorable cells",
};

const SCALE: Record<SmearResponse["scale_source"], string> = {
  red_cells_in_image: "measured from red cells in this image",
  um_per_px: "from the pixel size you entered",
  rbc_diameter_px: "from the red-cell size you entered",
  unknown: "unknown",
};

export function SmearAnalyser() {
  const inputId = useId();
  const umId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [um, setUm] = useState("");
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SmearResponse | null>(null);

  function choose(f: File | undefined | null) {
    setError(null);
    setResult(null);
    if (!f) return;
    const okType = ACCEPTED.includes(f.type) || /\.(png|jpe?g|bmp)$/i.test(f.name);
    if (!okType) return setError("Please choose a PNG, JPEG or BMP image.");
    if (f.size > SMEAR_MAX_BYTES) return setError("That image is larger than 10 MB. Please choose a smaller one.");
    setFile(f);
  }

  async function run(f: File) {
    setBusy(true);
    setError(null);
    setResult(null);
    const umVal = parseFloat(um);
    try {
      const um = umVal > 0 ? umVal : undefined;
      setResult(RUN_IN_BROWSER
        ? await import("@/lib/browserSmear").then((m) => m.analyzeSmearInBrowser(f, um, setStatus))
        : await analyzeSmear(f, um));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function trySample(path: string, label: string) {
    setError(null);
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error("Could not load the sample smear. Please try again.");
      const f = new File([await res.blob()], `${label.toLowerCase().replace(/\s+/g, "-")}.jpg`, { type: "image/jpeg" });
      setFile(f);
      await run(f);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const s = result?.summary;

  return (
    <>
      <section className="tray" aria-labelledby="smear-title">
        <div className="tray-head">
          <h2 id="smear-title" className="eyebrow">Smear tray · find and read every white cell</h2>
          <span className="label">PNG · JPEG · BMP · up to 10 MB</span>
        </div>

        <label
          htmlFor={inputId}
          className={`drop${drag ? " drag" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files?.[0]); }}
        >
          <span className="well" aria-hidden="true">
            <svg width="34" height="34" viewBox="0 0 34 34">
              <circle cx="9" cy="10" r="5" fill="none" stroke="#d4573a" strokeWidth="1.3" />
              <circle cx="24" cy="9" r="5" fill="none" stroke="#d4573a" strokeWidth="1.3" />
              <circle cx="11" cy="25" r="5" fill="none" stroke="#d4573a" strokeWidth="1.3" />
              <circle cx="23" cy="23" r="7" fill="#2b3a78" />
            </svg>
          </span>
          <span>
            <span className="big" style={{ display: "block" }}>{file ? file.name : "Drop a smear image here"}</span>
            <span className="small">or press to choose a file. A stained peripheral-blood field at 40x to 100x, with red cells visible.</span>
          </span>
          <input ref={inputRef} id={inputId} type="file" accept=".png,.jpg,.jpeg,.bmp,image/png,image/jpeg,image/bmp"
            onChange={(e) => choose(e.target.files?.[0])} />
        </label>

        <div className="field-row">
          <label htmlFor={umId} className="small">Pixel size in µm, if you know it (optional, makes the scale exact)</label>
          <input id={umId} className="num-input" inputMode="decimal" placeholder="e.g. 0.1" value={um}
            onChange={(e) => setUm(e.target.value.replace(/[^0-9.]/g, ""))} />
        </div>

        <div className="actions">
          <button type="button" className="btn btn-primary" disabled={!file || busy} onClick={() => file && run(file)}>
            {busy ? "Analysing..." : "Analyse smear"}
          </button>
          <span role="status" aria-live="polite" className="small">
            {!busy ? "" : RUN_IN_BROWSER ? status || "Starting" : "Finding cells, then scoring each one with the five-model ensemble."}
          </span>
        </div>

        {error && <p className="error" role="alert">{error}</p>}

        <div className="samples">
          <span className="small">No smear to hand? Try a healthy-donor smear from the BCCD dataset (MIT licence):</span>
          {SAMPLES.map((x) => (
            <button key={x.file} type="button" className="sample" disabled={busy} onClick={() => trySample(x.file, x.label)}>
              <img src={x.file} alt="" />
              {x.label}
            </button>
          ))}
        </div>
      </section>

      {result && s && (
        <article className="result-card smear-result" aria-labelledby="smear-verdict">
          <div className="result-head">
            <span className="label">Smear report · {result.model_version}</span>
            <span className="label">{result.latency_ms} ms</span>
          </div>
          <span className={`stamp ${s.smear_decision === "no_scorable_cells" ? "refer_to_expert" : s.smear_decision}`}>{LABEL[s.smear_decision]}</span>
          <h2 id="smear-verdict" className="display smear-h">
            {s.cells_found} white {s.cells_found === 1 ? "cell" : "cells"} found,
            <br />
            <em>{s.cells_scored} read by the model.</em>
          </h2>
          <p className="smear-note">
            The smear verdict applies the single-cell band to the mean score of the cells that were read. It is a research summary of
            these cells, not a probability that the person has leukemia.
          </p>

          <dl className="facts">
            <div><dt>Leukemic</dt><dd>{s.leukemic}</dd></div>
            <div><dt>Normal</dt><dd>{s.normal}</dd></div>
            <div><dt>Refer</dt><dd>{s.refer_to_expert}</dd></div>
            <div><dt>Not scored</dt><dd>{s.not_scored}</dd></div>
            <div><dt>Mean score</dt><dd>{s.mean_prob_calibrated == null ? "–" : s.mean_prob_calibrated.toFixed(3)}</dd></div>
            <div><dt>Scale</dt><dd className="smear-scale">{SCALE[result.scale_source]}</dd></div>
          </dl>

          {result.warnings.length > 0 && (
            <ul className="smear-warn">
              {result.warnings.map((w) => <li key={w}>{w}</li>)}
            </ul>
          )}

          <figure className="smear-figure">
            <img src={`data:image/png;base64,${result.overlay_png_base64}`}
              alt={`The smear with ${s.cells_found} white cells outlined and numbered; colours show each cell's result.`} />
            <figcaption className="scale-key">
              <span><i className="k-leuk" />leukemic</span>
              <span><i className="k-norm" />normal</span>
              <span><i className="k-ref" />refer to expert</span>
              <span><i className="k-ns" />not scored</span>
            </figcaption>
          </figure>

          {result.cells.length > 0 && (
            <div className="table-wrap smear-table">
              <table>
                <caption className="visually-hidden">Every white cell found in the smear</caption>
                <thead>
                  <tr><th scope="col">Cell</th><th scope="col">Result</th><th scope="col" className="num">Score</th><th scope="col">Note</th></tr>
                </thead>
                <tbody>
                  {result.cells.map((c) => (
                    <tr key={c.id}>
                      <th scope="row">{c.id}</th>
                      <td><span className={`dot d-${c.decision}`} aria-hidden="true" />{LABEL[c.decision]}</td>
                      <td className="num">{c.prob_calibrated == null ? "–" : c.prob_calibrated.toFixed(3)}</td>
                      <td className="small">{c.reason ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>
      )}
    </>
  );
}
