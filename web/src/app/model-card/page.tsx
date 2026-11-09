import type { Metadata } from "next";
import { getActiveModel } from "@/lib/activeModel";
import results from "@/data/results.json";

export const metadata: Metadata = { title: "Model card" };
export const revalidate = 300;

const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const ci = (c: number[] | null | undefined) => (c ? `${f3(c[0])} to ${f3(c[1])}` : "");

export default async function ModelCard() {
  const m = await getActiveModel();
  const d = results.data;
  const b = results.band;
  const e = results.explain;

  return (
    <div className="wrap section" style={{ paddingTop: 48 }}>
      <header className="center">
        <p className="eyebrow">Model card · {m.name}</p>
        <h1 className="display" style={{ marginTop: 12 }}>
          What it is,
          <br />
          <em>and what it is not.</em>
        </h1>
        <p className="lede" style={{ margin: "20px auto 0" }}>
          A research model that classifies single white blood cell images as acute lymphoblastic leukemia (ALL) or normal, with
          calibrated confidence and a refer-to-expert band. It is not a diagnostic device and has not been clinically validated.
        </p>
      </header>

      <section className="grid-2" style={{ marginTop: 48 }} aria-label="Summary">
        <div className="card">
          <h2 className="eyebrow">Model</h2>
          <table>
            <tbody>
              <tr><th scope="row">Architecture</th><td>{m.architecture}</td></tr>
              <tr><th scope="row">Pretraining</th><td>ImageNet, then two-stage fine-tuning (head, then last blocks)</td></tr>
              <tr><th scope="row">Calibration</th><td>Temperature {m.temperature.toFixed(4)} on the ensemble logit</td></tr>
              <tr><th scope="row">Referral band</th><td>[{m.t_low.toFixed(2)}, {m.t_high.toFixed(2)}], chosen for {m.band_target}</td></tr>
              <tr><th scope="row">Runtime</th><td>ONNX Runtime on CPU, {results.model.onnx_mb} MB, about 70 ms per cell with heatmap</td></tr>
            </tbody>
          </table>
          <p className="small" style={{ marginBottom: 0 }}>
            Settings read {m.source === "database" ? "live from the model registry" : "from the bundled export"}.
          </p>
        </div>
        <div className="card">
          <h2 className="eyebrow">Data</h2>
          <table>
            <tbody>
              <tr><th scope="row">Source</th><td>C-NMC 2019 (ISBI challenge), via TCIA, CC BY 3.0</td></tr>
              <tr><th scope="row">After audit</th><td>{d.images.toLocaleString("en-IN")} cells from {d.subjects} independent subjects</td></tr>
              <tr><th scope="row">Removed</th><td>{d.corrupt_removed} corrupt files, {d.duplicates_removed} pixel-identical copies</td></tr>
              <tr><th scope="row">Split</th><td>By subject, never by image: train {d.splits.train.subjects}, validation {d.splits.val.subjects}, test {d.splits.test.subjects} subjects</td></tr>
              <tr><th scope="row">ID aliasing</th><td>{d.merged_groups} groups of patient IDs shared identical images and were merged into one subject each</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section style={{ marginTop: 56 }} aria-labelledby="perf">
        <h2 id="perf" className="display">Performance on <em>held-out subjects.</em></h2>
        <p className="lede">
          All numbers below are on the validation subjects with 95% intervals from resampling whole subjects. The separate test set
          ({d.splits.test.subjects} subjects) is untouched and will be reported once, after every choice is frozen.
        </p>
        <div className="card table-wrap" style={{ marginTop: 20 }}>
          <table>
            <caption>Validation, threshold 0.5. Deployed model: ensemble of 5 EfficientNet-B0 seeds (PR-AUC {f3(results.headline.pr_auc)}, interval {ci(results.headline.pr_auc_ci)}).</caption>
            <thead>
              <tr><th scope="col">Model</th><th scope="col" className="num">PR-AUC</th><th scope="col" className="num">ROC-AUC</th><th scope="col" className="num">Recall (ALL)</th><th scope="col" className="num">Specificity</th><th scope="col" className="num">ECE</th></tr>
            </thead>
            <tbody>
              {results.models.map((r) => (
                <tr key={r.name} className={r.name.startsWith("EfficientNet") ? "hl" : ""}>
                  <th scope="row" style={{ textTransform: "none", fontSize: 14.5, color: "var(--ink)", letterSpacing: 0 }}>{r.name}</th>
                  <td className="num">{f3(r.pr_auc)} <span className="ci">{ci(r.pr_auc_ci)}</span></td>
                  <td className="num">{f3(r.roc_auc)} <span className="ci">{ci(r.roc_auc_ci)}</span></td>
                  <td className="num">{f3(r.recall)}</td>
                  <td className="num">{f3(r.specificity)}</td>
                  <td className="num">{f3(r.ece)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="grid-2" style={{ marginTop: 40 }} aria-label="Calibration and referral">
        <div className="card">
          <h2 className="eyebrow">Calibration</h2>
          <table>
            <thead><tr><th scope="col">Variant</th><th scope="col" className="num">ECE</th><th scope="col" className="num">NLL</th></tr></thead>
            <tbody>
              {results.calibration.map((c) => (
                <tr key={c.label}><td>{c.label}</td><td className="num">{c.ece.toFixed(4)}</td><td className="num">{c.nll.toFixed(4)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="small">Temperature fitted on two thirds of validation subjects and scored on the rest. Scaling helps a single model; the ensemble is already well calibrated.</p>
          <figure className="figure">
            <img src="/figures/reliability.png" alt="Reliability diagrams for a single model, MobileNetV3 and the five-seed ensemble, before and after temperature scaling. The ensemble curve lies closest to the diagonal." />
            <figcaption>Reliability on validation. The diagonal is perfect calibration.</figcaption>
          </figure>
        </div>
        <div className="card">
          <h2 className="eyebrow">Refer-to-expert band</h2>
          <table>
            <thead><tr><th scope="col">Policy</th><th scope="col" className="num">Referred</th><th scope="col" className="num">Missed ALL</th><th scope="col" className="num">False alarms</th></tr></thead>
            <tbody>
              <tr><td>Single threshold 0.5</td><td className="num">0%</td><td className="num">{pct(b.fixed.miss_rate)}</td><td className="num">{pct(b.fixed.false_alarm_rate)}</td></tr>
              <tr><td>Threshold lowered to {b.threshold_only.threshold.toFixed(2)}</td><td className="num">0%</td><td className="num">{pct(b.threshold_only.miss_rate)}</td><td className="num">{pct(b.threshold_only.false_alarm_rate)}</td></tr>
              <tr><td>Random referral</td><td className="num">{pct(b.random.referral_rate)}</td><td className="num">{pct(b.random.miss_rate)}</td><td className="num">n/a</td></tr>
              <tr className="hl"><td>Referral band</td><td className="num">{pct(b.band.referral_rate)}</td><td className="num">{pct(b.band.miss_rate)}</td><td className="num">{pct(b.band.false_alarm_rate)}</td></tr>
            </tbody>
          </table>
          <p className="small">
            Per 100 referred cells the band avoids {b.band.misses_avoided_per_100_referred.toFixed(1)} missed leukemic cells; random referral
            avoids {b.random.misses_avoided_per_100_referred.toFixed(1)}. Cross-fitted over subjects; the 2% target was met in-sample, not out of sample.
          </p>
          <figure className="figure">
            <img src="/figures/risk_coverage.png" alt="Risk-coverage curve: error rate on automatically decided cells rises as more cells are decided automatically." />
            <figcaption>Risk against coverage on validation.</figcaption>
          </figure>
        </div>
      </section>

      <section className="card" style={{ marginTop: 40 }} aria-labelledby="ext">
        <h2 id="ext" className="eyebrow">External and test results</h2>
        <p style={{ marginBottom: 0 }}>
          <span className="chip">Pending</span> Final test-set numbers and the external evaluation on ALL-IDB2 (a different lab and
          microscope) are not published yet. ALL-IDB2 access has been requested from the University of Milan; its images will never be shown
          here, only aggregate metrics.
        </p>
      </section>

      <section style={{ marginTop: 56 }} aria-labelledby="explain">
        <h2 id="explain" className="display">What the model <em>actually uses.</em></h2>
        <div className="grid-2" style={{ marginTop: 20 }}>
          <div className="card">
            <ul className="limits" style={{ paddingLeft: 18, margin: 0 }}>
              <li>Grad-CAM puts a median {e.gradcam_enrichment.toFixed(1)} times more weight on the cell than its area would predict, so the model looks at the cell, not the background.</li>
              <li>The heatmaps do not find specific features inside the cell better than chance: revealing random cell pixels restores the score as fast as following the heatmap (insertion {e.insertion.random_cell.toFixed(3)} vs {e.insertion.gradcam.toFixed(3)}).</li>
              <li>
                Cell size is a dominant cue. Among normal cells, size and leukemia score correlate at {e.size_rho_normal_seeds[0].toFixed(2)} to {e.size_rho_normal_seeds[1].toFixed(2)} across five seeds. Large normal cells are the main source of false alarms.
                {e.size_rho_scalejit != null && ` Scale-jitter augmentation only moved this to ${e.size_rho_scalejit.toFixed(2)}.`}
              </li>
            </ul>
          </div>
          <figure className="figure card">
            <img src="/figures/gallery_false_positive.png" alt="Five normal cells the model scored as leukemic, with Grad-CAM and Integrated Gradients heatmaps. Most are large cells." />
            <figcaption>Normal cells the model wrongly flagged, with their heatmaps.</figcaption>
          </figure>
        </div>
      </section>

      <section className="card" style={{ marginTop: 40 }} aria-labelledby="lim">
        <h2 id="lim" className="eyebrow">Limitations</h2>
        <ul className="limits" style={{ paddingLeft: 18 }}>
          <li>Single-cell crops from one dataset and one acquisition setup. Whole slides, other stains and other microscopes are out of scope.</li>
          <li>Only {d.subjects} independent subjects; intervals are wide and most differences between models are not established.</li>
          <li>Over-reliance on cell size, which depends on magnification. Expect a drop on images from other labs.</li>
          <li>Validation numbers only until the one-time test run is published.</li>
          <li>Not a medical device. Do not use it to make or support clinical decisions.</li>
        </ul>
        <p className="small" style={{ marginBottom: 0 }}>
          Data citation: Mourya S., Kant S., Kumar P., Gupta A., Gupta R. (2019). ALL Challenge dataset of ISBI 2019 (C-NMC 2019).
          The Cancer Imaging Archive. doi:10.7937/tcia.2019.dc64i46r
        </p>
      </section>
    </div>
  );
}
