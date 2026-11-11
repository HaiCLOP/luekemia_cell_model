import type { Metadata } from "next";
import { SmearAnalyser } from "@/components/SmearAnalyser";

export const metadata: Metadata = { title: "Smear" };

export default function SmearPage() {
  return (
    <div className="wrap section" style={{ paddingTop: 48 }}>
      <header style={{ maxWidth: 760 }}>
        <p className="eyebrow">Whole field · cell finder in front of the classifier</p>
        <h1 className="display" style={{ marginTop: 14 }}>
          A whole smear,
          <br />
          <em>one cell at a time.</em>
        </h1>
        <p className="lede" style={{ marginTop: 20 }}>
          Upload a stained blood-smear field. Each white cell is found by its stain, red cells are used as a ruler to bring every cell
          to the magnification the model was trained at, and each lymphoid-looking cell is cut out onto black and read on its own.
          Lobed cells such as neutrophils are outlined but not scored: the model only knows lymphocytes.
        </p>
      </header>
      <aside className="smear-caveat" aria-labelledby="caveat-title">
        <h2 id="caveat-title" className="eyebrow">Experimental · read before trusting a verdict</h2>
        <p>
          The cell finder works well: it found 99.7% of white cells on 364 BCCD smears. The classifier behind it does not travel.
          It learned from one lab&rsquo;s images. On 3,216 smears from another hospital it called two thirds of the benign smears
          leukemic (ROC-AUC 0.69), and on healthy BCCD smears it called 90% of the cells leukemic. Sample B below is a normal
          lymphocyte from a healthy donor, and you will see it marked leukemic. That is the failure this page exists to show, not a
          diagnosis. Full figures are in the smear section of the model card.
        </p>
      </aside>
      <div style={{ marginTop: 28 }}>
        <SmearAnalyser />
      </div>
    </div>
  );
}
