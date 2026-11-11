// Checks user-entered laboratory values against published diagnostic criteria for CLL and CML, and flags a blast
// count in the acute-leukemia range. Transparent rules, not a trained model: there is no public paired dataset to
// learn from. Research and education only; it never diagnoses, it says which criteria the given values meet and
// which test would settle the question.
//
// Sources
//  CLL: Hallek M et al. iwCLL guidelines for diagnosis, indications for treatment... of CLL. Blood 2018;131:2745-2760.
//  CLL immunophenotype score: Matutes E et al. Leukemia 1994;8:1640-1645 (CD5+, CD23+, FMC7-, weak sIg, weak CD79b).
//  CML and blast thresholds: Khoury JD et al. WHO Classification, 5th ed. Leukemia 2022;36:1703-1719;
//  Arber DA et al. International Consensus Classification (ICC). Blood 2022;140:1200-1228.

export type YesNo = "yes" | "no" | "unknown";
export type Marker = "positive" | "negative" | "dim" | "unknown";

export interface BloodInput {
  wbc?: number;                 // x10^9/L
  lymphocytesAbs?: number;      // x10^9/L
  lymphocytesPct?: number;      // % of WBC (used if the absolute count is missing)
  clonalBCells?: number;        // x10^9/L, from flow cytometry
  lightChainRestricted?: YesNo; // kappa or lambda restriction shown by flow cytometry
  cd5?: Marker;
  cd19?: Marker;
  cd20?: Marker;
  cd23?: Marker;
  fmc7?: Marker;
  cd79b?: Marker;
  sIg?: Marker;
  smudgeCells?: YesNo;
  lymphadenopathyOrOrganomegaly?: YesNo;
  basophilsPct?: number;        // % in blood
  blastsPct?: number;           // % in blood (or marrow, if that is what the user has)
  leftShift?: YesNo;            // myelocytes / metamyelocytes present in blood
  platelets?: number;           // x10^9/L
  splenomegaly?: YesNo;
  bcrAbl1?: "positive" | "negative" | "not_done";
}

export type Verdict = "meets" | "partly" | "not_suggested" | "cannot_assess";

export interface Finding {
  condition: "CLL" | "CML" | "Acute leukemia (blast count)";
  verdict: Verdict;
  headline: string;
  reasons: string[];
  nextStep: string;
}

const has = (v: number | undefined): v is number => typeof v === "number" && Number.isFinite(v);

export function absoluteLymphocytes(i: BloodInput): number | undefined {
  if (has(i.lymphocytesAbs)) return i.lymphocytesAbs;
  if (has(i.wbc) && has(i.lymphocytesPct)) return (i.wbc * i.lymphocytesPct) / 100;
  return undefined;
}

/** Matutes score: 1 point each for CD5+, CD23+, FMC7-, weak sIg, weak/negative CD79b. 4-5 is typical of CLL. */
export function matutesScore(i: BloodInput): { score: number; assessed: number } {
  const items: [Marker | undefined, (m: Marker) => boolean][] = [
    [i.cd5, (m) => m === "positive" || m === "dim"],
    [i.cd23, (m) => m === "positive" || m === "dim"],
    [i.fmc7, (m) => m === "negative"],
    [i.sIg, (m) => m === "dim" || m === "negative"],
    [i.cd79b, (m) => m === "dim" || m === "negative"],
  ];
  let score = 0, assessed = 0;
  for (const [m, point] of items) {
    if (!m || m === "unknown") continue;
    assessed++;
    if (point(m)) score++;
  }
  return { score, assessed };
}

export function checkCLL(i: BloodInput): Finding {
  const alc = absoluteLymphocytes(i);
  const reasons: string[] = [];
  const phenotypeKnown = [i.cd5, i.cd19, i.cd23].every((m) => m && m !== "unknown");
  const typical = phenotypeKnown && i.cd5 !== "negative" && i.cd19 !== "negative" && i.cd23 !== "negative";
  const { score, assessed } = matutesScore(i);
  const clonal = i.lightChainRestricted === "yes";

  if (has(i.clonalBCells)) {
    reasons.push(`Clonal B cells ${i.clonalBCells} x10^9/L (iwCLL threshold: at least 5).`);
    if (i.clonalBCells >= 5 && clonal && typical) {
      reasons.push("Clonality shown and the CD5+ CD19+ CD23+ pattern is present.");
      if (assessed >= 4) reasons.push(`Matutes score ${score} of ${assessed} markers (4-5 is typical of CLL).`);
      return { condition: "CLL", verdict: "meets", headline: "Values meet the iwCLL 2018 laboratory criteria for CLL",
        reasons, nextStep: "A haematologist confirms the diagnosis, including that the count has persisted for at least 3 months." };
    }
    if (i.clonalBCells < 5 && clonal) {
      const nodes = i.lymphadenopathyOrOrganomegaly === "yes";
      reasons.push(nodes
        ? "Below 5 x10^9/L but with enlarged lymph nodes or organs: small lymphocytic lymphoma (the tissue form of CLL) is possible."
        : "Below 5 x10^9/L without enlarged nodes or organs: this fits monoclonal B-cell lymphocytosis (MBL), a precursor, not CLL.");
      return { condition: "CLL", verdict: "partly", headline: nodes ? "Possible small lymphocytic lymphoma" : "Pattern of MBL, below the CLL threshold",
        reasons, nextStep: nodes ? "Lymph node or tissue assessment by a haematologist." : "Repeat counts over time; a haematologist decides on follow-up." };
    }
    if (i.clonalBCells >= 5 && (!clonal || !typical)) {
      reasons.push(!clonal ? "Clonality (light-chain restriction) not shown." : "The CD5+ CD19+ CD23+ pattern is not fully present.");
      return { condition: "CLL", verdict: "partly", headline: "Enough B cells, but the flow cytometry does not complete the CLL picture",
        reasons, nextStep: "A haematologist reviews the full flow-cytometry panel; other B-cell lymphomas can look similar." };
    }
  }
  if (has(alc)) {
    reasons.push(`Absolute lymphocytes ${alc.toFixed(1)} x10^9/L.`);
    if (alc >= 5) {
      if (i.smudgeCells === "yes") reasons.push("Smudge cells reported on the smear, common in CLL.");
      return { condition: "CLL", verdict: "partly", headline: "Lymphocytosis: CLL cannot be confirmed or excluded without flow cytometry",
        reasons, nextStep: "Flow cytometry of blood (CD5, CD19, CD20, CD23, light chains) decides whether this is CLL." };
    }
    reasons.push("Below 5 x10^9/L, so the blood count alone does not suggest CLL.");
    return { condition: "CLL", verdict: "not_suggested", headline: "Values given do not suggest CLL", reasons,
      nextStep: "Nothing further from these values; enlarged lymph nodes would still need assessment." };
  }
  return { condition: "CLL", verdict: "cannot_assess", headline: "Not enough information for CLL",
    reasons: ["Needs the lymphocyte count, or flow-cytometry results."], nextStep: "Enter WBC and lymphocyte % (or the absolute count)." };
}

export function checkCML(i: BloodInput): Finding {
  const reasons: string[] = [];
  const supportive: string[] = [];
  if (has(i.wbc) && i.wbc > 25) supportive.push(`raised white count (${i.wbc} x10^9/L)`);
  if (i.leftShift === "yes") supportive.push("myelocytes or metamyelocytes in blood (left shift)");
  if (has(i.basophilsPct) && i.basophilsPct >= 2) supportive.push(`basophilia (${i.basophilsPct}%)`);
  if (has(i.platelets) && i.platelets > 450) supportive.push(`raised platelets (${i.platelets} x10^9/L)`);
  if (i.splenomegaly === "yes") supportive.push("enlarged spleen");
  if (supportive.length) reasons.push(`Supporting features: ${supportive.join(", ")}.`);

  const phase = (): string | null => {
    if (has(i.blastsPct) && i.blastsPct >= 20) return "Blasts at or above 20%: blast phase (WHO 2022 and ICC 2022).";
    if ((has(i.blastsPct) && i.blastsPct >= 10) || (has(i.basophilsPct) && i.basophilsPct >= 20))
      return "Blasts 10-19% or basophils at least 20%: features of accelerated phase under ICC 2022.";
    return null;
  };

  if (i.bcrAbl1 === "positive") {
    reasons.push("BCR::ABL1 (Philadelphia chromosome) positive, the defining genetic finding of CML.");
    const p = phase();
    if (p) reasons.push(p);
    return { condition: "CML", verdict: "meets", headline: "Values meet the genetic criterion for CML (with a compatible blood picture)",
      reasons, nextStep: "A haematologist confirms the diagnosis and phase, usually with a bone-marrow examination." };
  }
  if (i.bcrAbl1 === "negative") {
    reasons.push("BCR::ABL1 negative: CML is excluded by definition.");
    return { condition: "CML", verdict: "not_suggested", headline: "CML excluded by a negative BCR::ABL1 test", reasons,
      nextStep: supportive.length >= 2 ? "Other myeloproliferative conditions can look similar; a haematologist decides." : "Nothing further for CML." };
  }
  if (supportive.length >= 2) {
    return { condition: "CML", verdict: "partly", headline: "Blood picture compatible with CML; the gene test is missing",
      reasons, nextStep: "BCR::ABL1 testing (PCR or FISH) decides whether this is CML." };
  }
  if (has(i.wbc)) {
    return { condition: "CML", verdict: "not_suggested", headline: "Values given do not suggest CML",
      reasons: reasons.length ? reasons : ["No raised white count, left shift or basophilia reported."], nextStep: "Nothing further from these values." };
  }
  return { condition: "CML", verdict: "cannot_assess", headline: "Not enough information for CML",
    reasons: ["Needs at least the white count, or a BCR::ABL1 result."], nextStep: "Enter WBC, basophil %, and whether myelocytes are present." };
}

export function checkBlasts(i: BloodInput): Finding {
  if (!has(i.blastsPct)) {
    return { condition: "Acute leukemia (blast count)", verdict: "cannot_assess", headline: "No blast count given",
      reasons: ["Acute leukemias (ALL, AML) are recognised mainly by the share of blasts."], nextStep: "Enter the blast % from the smear or marrow report if you have it." };
  }
  if (i.blastsPct >= 20)
    return { condition: "Acute leukemia (blast count)", verdict: "meets", headline: "Blast count in the acute-leukemia range (at least 20%)",
      reasons: [`Blasts ${i.blastsPct}%. WHO 2022 uses 20% for most acute leukemias; some genetic types qualify below that.`],
      nextStep: "Urgent haematology review. Flow cytometry tells ALL from AML; this site's Cell page reads only B-ALL blasts." };
  if (i.blastsPct >= 10)
    return { condition: "Acute leukemia (blast count)", verdict: "partly", headline: "Raised blasts (10-19%)",
      reasons: [`Blasts ${i.blastsPct}%. ICC 2022 accepts 10% for AML with certain genetic changes; also seen in other marrow disorders.`],
      nextStep: "Haematology review with genetic testing." };
  return { condition: "Acute leukemia (blast count)", verdict: i.blastsPct > 0 ? "partly" : "not_suggested",
    headline: i.blastsPct > 0 ? "A few blasts reported" : "No blasts reported",
    reasons: [`Blasts ${i.blastsPct}%. Any blasts in adult blood are unusual and worth a specialist look.`],
    nextStep: i.blastsPct > 0 ? "Repeat smear review by a haematologist." : "Nothing further from the blast count." };
}

export function checkAll(i: BloodInput): Finding[] {
  return [checkBlasts(i), checkCLL(i), checkCML(i)];
}
