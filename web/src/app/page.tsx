import Link from "next/link";
import { CellMark } from "@/components/CellMark";
import { Uploader } from "@/components/Uploader";
import results from "@/data/results.json";

const pct = (v: number, d = 1) => `${(100 * v).toFixed(d)}%`;

export default function Home() {
  const b = results.band;
  return (
    <>
      <section className="hero wrap">
        <p className="eyebrow">Acute lymphoblastic leukemia · single-cell microscopy</p>
        <h1 className="display" style={{ marginTop: 14 }}>
          Flag the blast cell.
          <br />
          <em>Know when to ask.</em>
        </h1>
        <p className="lede">
          Cell Triage looks at one white blood cell and says leukemic, normal, or refer to an expert. The third answer is
          the point: when the model is unsure it hands the cell to a person instead of guessing.
        </p>
        <CellMark className="hero-cell" />
        <Uploader />
      </section>

      <section className="wrap" aria-label="Key figures">
        <div className="ticket">
          <div>
            <div className="k">Cell images</div>
            <div className="v mono">{results.data.images.toLocaleString("en-IN")}</div>
          </div>
          <div>
            <div className="k">Independent subjects</div>
            <div className="v mono">{results.data.subjects}</div>
          </div>
          <div>
            <div className="k">Validation PR-AUC</div>
            <div className="v mono">{results.headline.pr_auc.toFixed(3)}</div>
          </div>
          <div>
            <div className="k">Missed leukemic cells</div>
            <div className="v mono">
              {pct(b.fixed.miss_rate)} → {pct(b.band.miss_rate)}
            </div>
          </div>
        </div>
        <p className="small center" style={{ marginTop: 12 }}>
          Validation figures from 18 held-out subjects. The last figure compares a single 0.5 threshold with the referral band,
          which sends {pct(b.band.referral_rate)} of cells to a person.
        </p>
      </section>

      <section className="section wrap">
        <h2 className="display center">
          From one cell
          <br />
          <em>to an honest answer.</em>
        </h2>
        <ol className="steps" style={{ listStyle: "none", padding: 0 }}>
          <li className="step">
            <div className="n">1</div>
            <h3>Upload a cell</h3>
            <p>A segmented white blood cell image, like the C-NMC 2019 crops the model was trained on.</p>
          </li>
          <li className="step">
            <div className="n">2</div>
            <h3>Five models vote</h3>
            <p>Five EfficientNet-B0 networks trained with different seeds score the cell; their average is calibrated.</p>
          </li>
          <li className="step">
            <div className="n">3</div>
            <h3>The band decides</h3>
            <p>
              Below {results.model.t_low.toFixed(2)} is normal, above {results.model.t_high.toFixed(2)} is leukemic, and anything in between
              goes to an expert.
            </p>
          </li>
          <li className="step">
            <div className="n">4</div>
            <h3>See the evidence</h3>
            <p>A heatmap shows which part of the image drove the score, and you can tell us if it got it wrong.</p>
          </li>
        </ol>
        <div className="actions" style={{ justifyContent: "center", marginTop: 44 }}>
          <Link href="/model-card" className="btn btn-primary">Read the model card</Link>
          <Link href="/method" className="btn btn-outline">How the evaluation works</Link>
        </div>
      </section>
    </>
  );
}
