"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import results from "@/data/results.json";
import { RESULT_KEY, sendFeedback, type StoredResult } from "@/lib/api";

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const BAND = results.band;

const COPY = {
  normal: {
    tag: "Normal",
    line1: "Looks like a",
    line2: "normal lymphocyte.",
    body: "The calibrated score sits below the lower edge of the band, where the model has been reliable on held-out subjects.",
  },
  leukemic: {
    tag: "Leukemic",
    line1: "Looks like a",
    line2: "leukemic blast.",
    body: "The calibrated score sits above the upper edge of the band. On validation subjects, scores this high were mostly lymphoblasts, but large normal cells can also land here.",
  },
  refer_to_expert: {
    tag: "Refer to expert",
    line1: "Not sure.",
    line2: "A person should look.",
    body: "The score falls inside the referral band. On validation subjects this band caught most of the cells the model would otherwise have missed, so the right next step is a human review, not a guess.",
  },
} as const;

export default function ResultPage() {
  const [data, setData] = useState<StoredResult | null | undefined>(undefined);
  const [label, setLabel] = useState<"leukemic" | "normal" | "unsure" | "">("");
  const [comment, setComment] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [msg, setMsg] = useState("");

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(RESULT_KEY);
      setData(raw ? (JSON.parse(raw) as StoredResult) : null);
    } catch {
      setData(null);
    }
  }, []);

  useEffect(() => {
    if (data) document.title = `${COPY[data.response.decision].tag} · Result · Cell Triage`;
  }, [data]);

  if (data === undefined) return <div className="prose section" aria-busy="true" />;

  if (data === null)
    return (
      <section className="prose section center">
        <h1 className="display" style={{ fontSize: "clamp(40px,6vw,64px)" }}>
          No result yet.
          <br />
          <em>Start with a cell.</em>
        </h1>
        <p className="lede" style={{ margin: "18px auto 0" }}>Results stay in this browser tab only. Upload an image or try a sample cell.</p>
        <div className="actions" style={{ justifyContent: "center" }}>
          <Link className="btn btn-primary" href="/#analyse">Analyse a cell</Link>
        </div>
      </section>
    );

  const r = data.response;
  const c = COPY[r.decision];
  const lo = r.band.t_low * 100, hi = r.band.t_high * 100, p = r.prob_calibrated * 100;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!label) return;
    setState("sending");
    try {
      await sendFeedback(r.prediction_id, label, comment);
      setState("sent");
      setMsg("Thank you. Your label is saved against this prediction.");
    } catch (err) {
      setState("error");
      setMsg((err as Error).message);
    }
  }

  return (
    <div className="prose section" style={{ paddingTop: 40 }}>
      <Link href="/#analyse" className="small">Analyse another cell</Link>

      <section className={`verdict ${r.decision}`} aria-labelledby="verdict-title">
        <span className="tag">{c.tag}</span>
        <h1 id="verdict-title" className="display">
          {c.line1}
          <br />
          <em>{c.line2}</em>
        </h1>
        <p style={{ marginTop: 16, maxWidth: 620 }}>{c.body}</p>

        <div className="pbar">
          <div className="small" id="pbar-label">
            Calibrated probability of leukemia: <strong className="mono">{r.prob_calibrated.toFixed(3)}</strong>
          </div>
          <div
            className="pbar-track"
            role="img"
            aria-labelledby="pbar-label"
            aria-describedby="pbar-legend"
            style={{ ["--lo" as string]: `${lo}%`, ["--hi" as string]: `${hi}%`, marginTop: 10 }}
          >
            <span className="pbar-mark" style={{ left: `${p}%` }} />
          </div>
          <div className="pbar-scale mono" aria-hidden="true"><span>0</span><span>0.5</span><span>1</span></div>
          <div className="pbar-legend" id="pbar-legend">
            <span><i style={{ background: "#cfe8bb" }} />normal, below {r.band.t_low.toFixed(2)}</span>
            <span><i style={{ background: "#f3dca0" }} />refer, {r.band.t_low.toFixed(2)} to {r.band.t_high.toFixed(2)}</span>
            <span><i style={{ background: "#ecc2b5" }} />leukemic, above {r.band.t_high.toFixed(2)}</span>
          </div>
        </div>

        <dl className="facts">
          <div><dt>Raw ensemble score</dt><dd className="mono">{r.prob_raw.toFixed(3)}</dd></div>
          <div><dt>Model</dt><dd className="mono">{r.model_version}</dd></div>
          <div><dt>Analysis time</dt><dd className="mono">{r.latency_ms} ms</dd></div>
          <div><dt>Image kept</dt><dd>{r.image_stored ? "Yes, 24 hours" : "No, fingerprint only"}</dd></div>
        </dl>
      </section>

      <div className="pair">
        <figure>
          <img src={data.previewDataUrl} alt={`The uploaded cell image, ${data.fileName}`} />
          <figcaption>Your image ({data.fileName})</figcaption>
        </figure>
        <figure>
          {r.heatmap_png_base64 ? (
            <img src={`data:image/png;base64,${r.heatmap_png_base64}`} alt="Heatmap over the cell: brighter areas pushed the score towards leukemic" />
          ) : (
            <p className="small">No heatmap returned.</p>
          )}
          <figcaption>Where the model looked. Brighter means more influence on the leukemic score.</figcaption>
        </figure>
      </div>

      <section className="note" aria-labelledby="refer-title">
        <h2 id="refer-title" style={{ fontSize: 19, margin: 0 }}>What “refer to expert” means</h2>
        <p style={{ marginBottom: 0 }}>
          The model has three answers, not two. Scores between {r.band.t_low.toFixed(2)} and {r.band.t_high.toFixed(2)} are where it has been
          wrong most often, so it hands those cells to a haematologist instead of guessing. On 18 held-out subjects this sent {pct(BAND.band.referral_rate)} of
          cells to review and cut missed leukemic cells from {pct(BAND.fixed.miss_rate)} to {pct(BAND.band.miss_rate)}. The heatmap is a guide to what the model used, not proof
          that it used the right thing: in testing it leaned heavily on cell size.
        </p>
      </section>

      <form className="note feedback" onSubmit={submit} aria-labelledby="fb-title">
        <h2 id="fb-title" style={{ fontSize: 19, margin: 0 }}>Did it get this one right?</h2>
        <fieldset disabled={state === "sending" || state === "sent"}>
          <legend className="small" style={{ marginTop: 6 }}>If you know the true label, tell us. It is saved with this prediction only.</legend>
          <div className="choice">
            {(["leukemic", "normal", "unsure"] as const).map((v) => (
              <label key={v}>
                <input type="radio" name="label" value={v} checked={label === v} onChange={() => setLabel(v)} />
                {v === "unsure" ? "Not sure" : v === "leukemic" ? "It is leukemic" : "It is normal"}
              </label>
            ))}
          </div>
          <label htmlFor="fb-comment" className="visually-hidden">Optional comment</label>
          <textarea
            id="fb-comment"
            maxLength={500}
            placeholder="Optional note, up to 500 characters"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
          <div className="actions">
            <button type="submit" className="btn btn-primary" disabled={!label}>
              {state === "sending" ? "Saving..." : "Send feedback"}
            </button>
            <span className="small">{comment.length}/500</span>
          </div>
        </fieldset>
        {msg && (
          <p role="status" className={state === "error" ? "error" : "small"} style={{ marginBottom: 0 }}>
            {msg}
          </p>
        )}
      </form>
    </div>
  );
}
