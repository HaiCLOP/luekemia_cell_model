import Link from "next/link";
import { Reticle } from "@/components/CellMark";
import { Uploader } from "@/components/Uploader";
import results from "@/data/results.json";

const pct = (v: number, d = 1) => `${(100 * v).toFixed(d)}%`;

export default function Home() {
  const d = results.deployed;
  const b = d.band_out_of_lab;
  const idb = d.all_idb2!;
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
              <div><span className="label">Unseen-lab ROC-AUC</span><b>{idb.roc_auc.toFixed(3)}</b></div>
              <div><span className="label">Training labs</span><b>{results.model.training_labs.length}</b></div>
              <div><span className="label">Model</span><b>{results.model.name}</b></div>
            </div>
          </div>

          <figure className="scope" style={{ margin: 0 }}>
            <div className="scope-field">
              <img src="/samples/sample-leukemic.png" alt="A lymphoblast from the C-NMC 2019 validation set, seen through a microscope field of view" />
            </div>
            <Reticle className="scope-reticle" />
            <figcaption className="scope-caption">
              C-NMC 2019 · subject 22 · <b>ALL</b>

            </figcaption>
          </figure>
        </section>

        <Uploader />

        <section className="report" aria-label="Key results">
          <div>
            <span className="label">ALL-IDB2, a lab never seen</span>
            <div className="v">{idb.roc_auc.toFixed(3)}</div>
            <div className="s">ROC-AUC on {idb.cells_cut_out} cells (95% CI {idb.roc_auc_ci95[0].toFixed(2)} to {idb.roc_auc_ci95[1].toFixed(2)})</div>
          </div>
          <div>
            <span className="label">Another hospital&apos;s smears</span>
            <div className="v">{d.taleqani_smear_roc_auc.toFixed(3)}</div>
            <div className="s">ROC-AUC per smear (Taleqani ALL); {d.taleqani_smear_roc_auc_retired.toFixed(3)} for the C-NMC-only MH51401122</div>
          </div>
          <div>
            <span className="label">Leave one lab out</span>
            <div className="v">{Math.min(...Object.values(d.leave_one_lab_out_roc_auc)).toFixed(2)}–{Math.max(...Object.values(d.leave_one_lab_out_roc_auc)).toFixed(2)}</div>
            <div className="s">ROC-AUC on each lab when trained on the other two</div>
          </div>
          <div>
            <span className="label">Missed leukemic cells</span>
            <div className="v">{pct(b.miss_rate)}</div>
            <div className="s">with {pct(b.referral_rate)} of cells referred, out of lab</div>
          </div>
        </section>
        <p className="small" style={{ marginTop: 12 }}>
          Every figure comes from cells of a lab the classifier was not fitted on. The referral band is set so that, out of lab, no
          more than 2% of leukemic cells are called normal; the price is that about half of all cells go to a person. ALL-IDB2
          intervals resample images, because the dataset publishes no patient IDs.
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
            <p>Upload one white blood cell cut out on a black background, or a whole smear on the Smear page.</p>
          </li>
          <li>
            <span className="no">STEP 02</span>
            <h3>The model reads it</h3>
            <p>A blood-cell foundation model describes the cell. A filter checks it is a lymphocyte or blast, then a classifier trained on three labs scores it.</p>
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
            <p>The model card shows how it did on labs it never saw, and you can tell us when it got a cell wrong.</p>
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
