import Link from "next/link";
import { Reticle } from "@/components/CellMark";
import { Uploader } from "@/components/Uploader";
import results from "@/data/results.json";

const pct = (v: number, d = 1) => `${(100 * v).toFixed(d)}%`;

export default function Home() {
  const b = results.band;
  return (
    <>
      <div className="wrap">
        <section className="hero" aria-labelledby="hero-title">
          <div>
            <p className="eyebrow">Acute lymphoblastic leukemia · single-cell microscopy</p>
            <h1 id="hero-title" className="display" style={{ marginTop: 18 }}>
              Flag the blast.
              <br />
              <em>Know when to ask.</em>
            </h1>
            <p className="lede">
              Cell Triage reads one white blood cell and answers leukemic, normal, or refer to an expert. The third answer is the
              point: when the model is unsure, it hands the cell to a person instead of guessing.
            </p>
            <div className="hero-meta">
              <div><span className="label">Validation PR-AUC</span><b>{results.headline.pr_auc.toFixed(3)}</b></div>
              <div><span className="label">Subjects</span><b>{results.data.subjects}</b></div>
              <div><span className="label">Per cell</span><b>~65 ms</b></div>
            </div>
          </div>

          <figure className="scope" style={{ margin: 0 }}>
            <div className="scope-field">
              <img src="/samples/sample-leukemic.png" alt="A lymphoblast from the C-NMC 2019 validation set, seen through a microscope field of view" />
            </div>
            <Reticle className="scope-reticle" />
            <figcaption className="scope-caption">
              C-NMC 2019 · subject 22 · <b>ALL</b>
              <br />
              model: leukemic, p = 0.72
            </figcaption>
          </figure>
        </section>

        <Uploader />

        <section className="report" aria-label="Key results">
          <div>
            <span className="label">Cells audited</span>
            <div className="v">{results.data.images.toLocaleString("en-IN")}</div>
            <div className="s">{results.data.corrupt_removed} corrupt and {results.data.duplicates_removed} duplicates removed</div>
          </div>
          <div>
            <span className="label">Independent subjects</span>
            <div className="v">{results.data.subjects}</div>
            <div className="s">split by person, never by image</div>
          </div>
          <div>
            <span className="label">Calibration error (ECE)</span>
            <div className="v">{results.headline.ece_calibrated.toFixed(3)}</div>
            <div className="s">deployed, temperature scaled; {results.headline.ece_raw.toFixed(3)} before scaling</div>
          </div>
          <div>
            <span className="label">Missed leukemic cells</span>
            <div className="v">
              {pct(b.fixed.miss_rate)}<span className="arr" aria-label="to">→</span><span className="to">{pct(b.band.miss_rate)}</span>
            </div>
            <div className="s">with {pct(b.band.referral_rate)} of cells referred</div>
          </div>
        </section>
        <p className="small" style={{ marginTop: 12 }}>
          Validation results on 18 subjects not used for training; the referral figures are nested cross-fitted over those subjects.
          On the final test, {results.test?.subjects} more subjects evaluated once, PR-AUC was {results.test?.pr_auc.toFixed(3)} and the band missed{" "}
          {results.test ? pct(results.test.band.miss_rate) : ""} of leukemic cells.
        </p>
      </div>

      <section className="wrap section">
        <p className="eyebrow">Protocol</p>
        <h2 className="display" style={{ marginTop: 14, maxWidth: 760 }}>
          From one cell <em>to an honest answer.</em>
        </h2>
        <ol className="protocol">
          <li>
            <span className="no">STEP 01</span>
            <h3>Mount the cell</h3>
            <p>Upload a segmented white blood cell image, like the C-NMC 2019 crops the model learned from.</p>
          </li>
          <li>
            <span className="no">STEP 02</span>
            <h3>Five models read it</h3>
            <p>Five EfficientNet-B0 networks trained from different seeds score the cell. Their average is calibrated so 0.8 means about 80%.</p>
          </li>
          <li>
            <span className="no">STEP 03</span>
            <h3>The band decides</h3>
            <p>
              Below {results.model.t_low.toFixed(2)} reads normal, above {results.model.t_high.toFixed(2)} reads leukemic. Anything in between goes to a
              haematologist.
            </p>
          </li>
          <li>
            <span className="no">STEP 04</span>
            <h3>Check the evidence</h3>
            <p>A heatmap shows where the model looked, and you can tell us when it got a cell wrong.</p>
          </li>
        </ol>
        <div className="actions" style={{ marginTop: 32 }}>
          <Link href="/model-card" className="btn btn-primary">Read the model card <span className="arrow">→</span></Link>
          <Link href="/method" className="btn btn-outline">How it was evaluated</Link>
        </div>
      </section>
    </>
  );
}
