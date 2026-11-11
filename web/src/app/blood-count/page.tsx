import type { Metadata } from "next";
import { BloodCountForm } from "@/components/BloodCountForm";

export const metadata: Metadata = { title: "Blood count" };

export default function BloodCountPage() {
  return (
    <div className="wrap section" style={{ paddingTop: 48 }}>
      <header style={{ maxWidth: 780 }}>
        <p className="eyebrow">CLL · CML · blast count · from your lab report</p>
        <h1 className="display" style={{ marginTop: 14 }}>
          The leukemias a cell
          <br />
          <em>picture cannot show.</em>
        </h1>
        <p className="lede" style={{ marginTop: 20 }}>
          CLL cells look almost like normal lymphocytes, and CML is a pattern across the whole blood count confirmed by a gene
          test. So instead of an image, this page takes the numbers from a lab report and checks them against the published
          diagnostic criteria. It is a transparent rule check, not a trained model, and it tells you which test would settle
          each question.
        </p>
      </header>

      <aside className="smear-caveat" aria-labelledby="bc-caveat">
        <h2 id="bc-caveat" className="eyebrow">Read first</h2>
        <p>
          Research and education only. Meeting or missing a criterion here is not a diagnosis: a haematologist weighs the full
          history, the smear, the marrow and repeat tests. Thresholds are for adults. Criteria: Hallek et al., iwCLL 2018
          (Blood 131:2745); Matutes et al. 1994 (Leukemia 8:1640); Khoury et al., WHO 5th edition 2022 (Leukemia 36:1703);
          Arber et al., ICC 2022 (Blood 140:1200).
        </p>
      </aside>

      <div style={{ marginTop: 28 }}>
        <BloodCountForm />
      </div>
    </div>
  );
}
