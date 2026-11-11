"use client";

import { useId, useState } from "react";
import { checkAll, type BloodInput, type Finding, type Marker, type YesNo } from "@/lib/bloodCriteria";

type NumKey = "wbc" | "lymphocytesAbs" | "lymphocytesPct" | "clonalBCells" | "basophilsPct" | "blastsPct" | "platelets";
type MarkerKey = "cd5" | "cd19" | "cd20" | "cd23" | "fmc7" | "cd79b" | "sIg";
type YesNoKey = "lightChainRestricted" | "smudgeCells" | "lymphadenopathyOrOrganomegaly" | "leftShift" | "splenomegaly";

const NUMBERS: { key: NumKey; label: string; unit: string }[] = [
  { key: "wbc", label: "White cell count (WBC)", unit: "x10⁹/L" },
  { key: "lymphocytesPct", label: "Lymphocytes", unit: "% of WBC" },
  { key: "lymphocytesAbs", label: "Lymphocytes, absolute (if reported)", unit: "x10⁹/L" },
  { key: "blastsPct", label: "Blasts", unit: "%" },
  { key: "basophilsPct", label: "Basophils", unit: "%" },
  { key: "platelets", label: "Platelets", unit: "x10⁹/L" },
  { key: "clonalBCells", label: "Clonal B cells (flow cytometry)", unit: "x10⁹/L" },
];
const MARKERS: { key: MarkerKey; label: string }[] = [
  { key: "cd5", label: "CD5" }, { key: "cd19", label: "CD19" }, { key: "cd20", label: "CD20" }, { key: "cd23", label: "CD23" },
  { key: "fmc7", label: "FMC7" }, { key: "cd79b", label: "CD79b" }, { key: "sIg", label: "Surface Ig" },
];
const YESNO: { key: YesNoKey; label: string }[] = [
  { key: "lightChainRestricted", label: "Light-chain restriction (kappa or lambda) on flow cytometry" },
  { key: "smudgeCells", label: "Smudge cells reported on the smear" },
  { key: "leftShift", label: "Myelocytes or metamyelocytes in blood (left shift)" },
  { key: "splenomegaly", label: "Enlarged spleen" },
  { key: "lymphadenopathyOrOrganomegaly", label: "Enlarged lymph nodes, liver or spleen" },
];

const VERDICT: Record<Finding["verdict"], string> = {
  meets: "Meets criteria",
  partly: "Partly / needs a test",
  not_suggested: "Not suggested",
  cannot_assess: "Cannot assess",
};

export function BloodCountForm() {
  const id = useId();
  const [nums, setNums] = useState<Partial<Record<NumKey, string>>>({});
  const [markers, setMarkers] = useState<Partial<Record<MarkerKey, Marker>>>({});
  const [yesno, setYesno] = useState<Partial<Record<YesNoKey, YesNo>>>({});
  const [bcr, setBcr] = useState<BloodInput["bcrAbl1"]>("not_done");
  const [findings, setFindings] = useState<Finding[] | null>(null);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const input: BloodInput = { bcrAbl1: bcr, ...markers, ...yesno };
    for (const { key } of NUMBERS) {
      const v = parseFloat(nums[key] ?? "");
      if (Number.isFinite(v) && v >= 0) input[key] = v;
    }
    setFindings(checkAll(input));
  }

  return (
    <>
      <form className="tray" onSubmit={submit} aria-labelledby={`${id}-t`}>
        <div className="tray-head">
          <h2 id={`${id}-t`} className="eyebrow">Report values · fill in what you have, leave the rest</h2>
          <span className="label">Adult reference · nothing leaves this browser</span>
        </div>

        <fieldset className="bc-group">
          <legend className="label">Blood count</legend>
          <div className="bc-grid">
            {NUMBERS.map(({ key, label, unit }) => (
              <label key={key} className="bc-field">
                <span>{label}</span>
                <span className="bc-input">
                  <input inputMode="decimal" className="num-input" value={nums[key] ?? ""}
                    onChange={(e) => setNums({ ...nums, [key]: e.target.value.replace(/[^0-9.]/g, "") })} />
                  <span className="small">{unit}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="bc-group">
          <legend className="label">Flow cytometry markers on the abnormal B cells</legend>
          <div className="bc-grid">
            {MARKERS.map(({ key, label }) => (
              <label key={key} className="bc-field">
                <span>{label}</span>
                <select className="num-input" value={markers[key] ?? "unknown"}
                  onChange={(e) => setMarkers({ ...markers, [key]: e.target.value as Marker })}>
                  <option value="unknown">not tested</option>
                  <option value="positive">positive</option>
                  <option value="dim">dim / weak</option>
                  <option value="negative">negative</option>
                </select>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="bc-group">
          <legend className="label">Findings</legend>
          <div className="bc-grid">
            {YESNO.map(({ key, label }) => (
              <label key={key} className="bc-field">
                <span>{label}</span>
                <select className="num-input" value={yesno[key] ?? "unknown"}
                  onChange={(e) => setYesno({ ...yesno, [key]: e.target.value as YesNo })}>
                  <option value="unknown">unknown</option>
                  <option value="yes">yes</option>
                  <option value="no">no</option>
                </select>
              </label>
            ))}
            <label className="bc-field">
              <span>BCR::ABL1 (Philadelphia chromosome) test</span>
              <select className="num-input" value={bcr} onChange={(e) => setBcr(e.target.value as BloodInput["bcrAbl1"])}>
                <option value="not_done">not done</option>
                <option value="positive">positive</option>
                <option value="negative">negative</option>
              </select>
            </label>
          </div>
        </fieldset>

        <div className="actions">
          <button type="submit" className="btn btn-primary">Check against the criteria</button>
          <span className="small">Rules from iwCLL 2018, WHO 2022 and ICC 2022. Not a diagnosis.</span>
        </div>
      </form>

      {findings && (
        <section aria-live="polite" aria-label="Results" className="bc-results">
          {findings.map((f) => (
            <article key={f.condition} className={`card bc-card v-${f.verdict}`}>
              <div className="bc-card-head">
                <h3>{f.condition}</h3>
                <span className="chip">{VERDICT[f.verdict]}</span>
              </div>
              <p className="bc-headline">{f.headline}</p>
              <ul className="limits">
                {f.reasons.map((r) => <li key={r}>{r}</li>)}
              </ul>
              <p className="small" style={{ marginBottom: 0 }}><b>Next step:</b> {f.nextStep}</p>
            </article>
          ))}
        </section>
      )}
    </>
  );
}
