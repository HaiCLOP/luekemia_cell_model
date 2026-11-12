import type { Metadata } from "next";
import { SmearAnalyser } from "@/components/SmearAnalyser";
import { RUN_IN_BROWSER } from "@/lib/api";

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
          It learned from one lab&rsquo;s images. A second network now skips neutrophils and other non-lymphocytes, and on the
          lymphocytes that remain, smears from another hospital are separated barely better than chance (ROC-AUC 0.57; 79% of
          benign smears called leukemic). A retrained, blur-tolerant version did worse, so it is not used. The cell-type filter
          has the same weakness: sample B below is a normal lymphocyte from a healthy donor, and the filter wrongly skips it, so
          you will get &ldquo;no scorable cells&rdquo;. Expect refusals and wrong answers on smears from other labs; that is what
          this page exists to show. Full figures are in the smear section of the model card.
        </p>
      </aside>
      {RUN_IN_BROWSER && (
        <p className="small" style={{ marginTop: 18, maxWidth: 780 }}>
          Everything runs inside your browser, including the Python cell finder (via Pyodide). The first smear downloads
          about 50 MB of Python tools and the two models; after that they are cached. Nothing is uploaded.
        </p>
      )}
      <div style={{ marginTop: 28 }}>
        <SmearAnalyser />
      </div>
    </div>
  );
}
