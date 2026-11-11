import type { Metadata } from "next";
import Link from "next/link";
import results from "@/data/results.json";

export const metadata: Metadata = { title: "Method" };

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

export default function Method() {
  const d = results.data;
  const b = results.band;
  return (
    <article className="prose section" style={{ paddingTop: 48 }}>
      <p className="eyebrow">Method</p>
      <h1 className="display" style={{ marginTop: 12, fontSize: "clamp(42px,6.5vw,76px)" }}>
        Split by person.
        <br />
        <em>Refer when unsure.</em>
      </h1>
      <p className="lede" style={{ marginTop: 20 }}>
        Two choices matter more than the network: how the data is split, and what the model does when it is not confident.
      </p>

      <section style={{ marginTop: 48 }} aria-labelledby="split">
        <h2 id="split" className="display" style={{ fontSize: "clamp(30px,4vw,42px)" }}>
          1. Patient-level splitting
        </h2>
        <p>
          Cells from the same person look alike: the same stain batch, the same smear, often the same disease stage. If one person&apos;s
          cells appear in both training and testing, a model can score well by recognising the person rather than the disease. Some
          published results on this dataset report 98 to 99% accuracy using image-level splits, which allow exactly this leak.
        </p>
        <p>
          Here every subject is assigned to one role only: {d.splits.train.subjects} subjects for training, {d.splits.val.subjects} for
          validation (choosing epochs, temperature and the band) and {d.splits.test.subjects} for a single final test. An automated test fails
          the build if any subject appears in two roles.
        </p>
        <div className="note">
          <h3>The audit found more than expected</h3>
          <p style={{ margin: 0 }}>
            The public copy contained {d.corrupt_removed} unreadable files and {d.duplicates_removed} pixel-identical duplicates. Worse,{" "}
            {d.merged_groups} groups of different patient IDs shared identical images, so they are almost certainly the same people recorded
            twice. Splitting by ID alone would have leaked them across roles. Those IDs were merged before splitting, leaving{" "}
            {d.subjects} independent subjects.
          </p>
        </div>
      </section>

      <section style={{ marginTop: 56 }} aria-labelledby="band">
        <h2 id="band" className="display" style={{ fontSize: "clamp(30px,4vw,42px)" }}>
          2. The referral band
        </h2>
        <p>
          A single threshold forces a yes or no on every cell. Lowering it to miss fewer leukemic cells floods the normal cells with false
          alarms: reaching a 2% miss rate that way raises false alarms to {pct(b.threshold_only.false_alarm_rate)}.
        </p>
        <p>
          Instead the model gets a third answer. Its scores are first calibrated so that 0.8 means roughly 80%. Then two thresholds are chosen
          on validation subjects: below the lower one the cell is called normal, above the upper one leukemic, and in between it goes to a
          person. Fitted on two thirds of the validation subjects and scored on the remaining third (rotated, nested), the band referred {pct(b.band.referral_rate)} of cells and cut missed leukemic cells from{" "}
          {pct(b.fixed.miss_rate)} to {pct(b.band.miss_rate)}, while false alarms stayed at {pct(b.band.false_alarm_rate)}. Referring the same
          share of cells at random barely helps ({pct(b.random.miss_rate)} missed).
        </p>
        <figure className="figure" style={{ marginTop: 20 }}>
          <img src="/figures/referral_band.png" alt="Histogram of calibrated scores for normal and leukemic validation cells, with the referral band shaded between 0.2 and 0.5." />
          <figcaption>Calibrated scores on validation. The shaded band is sent to an expert. Note the normal cells with high scores: no band can catch confident mistakes.</figcaption>
        </figure>
      </section>

      <section style={{ marginTop: 56 }} aria-labelledby="honest">
        <h2 id="honest" className="display" style={{ fontSize: "clamp(30px,4vw,42px)" }}>
          3. Keeping the numbers honest
        </h2>
        <ul className="limits" style={{ paddingLeft: 18 }}>
          <li>Every interval comes from resampling whole subjects, because cells from one person are not independent.</li>
          <li>Thresholds and temperature are fitted on part of the validation subjects and scored on the rest.</li>
          <li>The test set was evaluated once, after every choice was frozen, with the date, code version and model fingerprint recorded.</li>
          <li>An external dataset from another lab is used only for evaluation, never for training or tuning.</li>
        </ul>
        <div className="actions">
          <Link href="/model-card" className="btn btn-primary">See the full model card</Link>
          <Link href="/#analyse" className="btn btn-outline">Try a cell</Link>
        </div>
      </section>
    </article>
  );
}
