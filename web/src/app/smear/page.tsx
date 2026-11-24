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
          to the magnification the model was trained at, and every cell is cut out onto black. The model&apos;s own filter then sets
          aside cells that are not lymphocytes or blasts, such as neutrophils, and reads the rest one by one.
        </p>
      </header>
      <aside className="smear-caveat" aria-labelledby="caveat-title">
        <h2 id="caveat-title" className="eyebrow">Experimental · read before trusting a verdict</h2>
        <p>
          The cell finder works well: it found 99.7% of white cells on 364 BCCD smears. The hard part is reading the cells on
          smears from a lab the model never saw. The model, MH51401102, was trained on three labs (New Delhi, Johns Hopkins,
          Munich). On another hospital&rsquo;s leukemia smears it reaches ROC-AUC 0.83 (the retired single-lab model managed
          0.57), and it calls 26% of healthy-donor lymphocytes leukemic. Sample B is one of those 26%: the model wrongly calls
          it leukemic (score 0.55, just over the line). That is better, not good. Its scores run low on a lab it has not seen,
          so a fixed cut-off misses most leukemia smears there; a new lab would need to set its own cut-off on a few smears it
          has already diagnosed. Expect refusals and wrong answers; that is what this
          page exists to show. Full figures are in the smear section of the model card.
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
