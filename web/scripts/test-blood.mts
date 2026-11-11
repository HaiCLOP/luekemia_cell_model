// Tests for the CLL / CML / blast-count criteria checker. Run: node web/scripts/test-blood.mts (Node 22.6+ strips types).
import assert from "node:assert/strict";
import { absoluteLymphocytes, checkBlasts, checkCLL, checkCML, matutesScore } from "../src/lib/bloodCriteria.ts";

const cases: [string, () => void][] = [
  ["lymphocyte % converts to an absolute count", () => assert.equal(absoluteLymphocytes({ wbc: 20, lymphocytesPct: 50 }), 10)],
  ["classic CLL flow result meets criteria", () => {
    const f = checkCLL({ clonalBCells: 12, lightChainRestricted: "yes", cd5: "positive", cd19: "positive", cd23: "positive" });
    assert.equal(f.verdict, "meets");
  }],
  ["clonal B cells below 5 without nodes is MBL, not CLL", () => {
    const f = checkCLL({ clonalBCells: 3, lightChainRestricted: "yes", cd5: "positive", cd19: "positive", cd23: "positive" });
    assert.equal(f.verdict, "partly");
    assert.match(f.headline, /MBL/);
  }],
  ["lymphocytosis without flow asks for flow cytometry", () => {
    const f = checkCLL({ wbc: 30, lymphocytesPct: 80 });
    assert.equal(f.verdict, "partly");
    assert.match(f.nextStep, /Flow cytometry/);
  }],
  ["normal lymphocytes do not suggest CLL", () => assert.equal(checkCLL({ wbc: 7, lymphocytesPct: 30 }).verdict, "not_suggested")],
  ["CD23-negative clone does not complete the CLL picture", () => {
    const f = checkCLL({ clonalBCells: 8, lightChainRestricted: "yes", cd5: "positive", cd19: "positive", cd23: "negative" });
    assert.equal(f.verdict, "partly");
  }],
  ["Matutes score counts only assessed markers", () => assert.deepEqual(matutesScore({ cd5: "positive", cd23: "positive", fmc7: "negative" }), { score: 3, assessed: 3 })],
  ["BCR::ABL1 positive meets the CML genetic criterion", () => assert.equal(checkCML({ bcrAbl1: "positive", wbc: 120 }).verdict, "meets")],
  ["BCR::ABL1 negative excludes CML", () => assert.equal(checkCML({ bcrAbl1: "negative", wbc: 120, leftShift: "yes" }).verdict, "not_suggested")],
  ["CML-like blood picture without the gene test asks for it", () => {
    const f = checkCML({ wbc: 90, leftShift: "yes", basophilsPct: 6, bcrAbl1: "not_done" });
    assert.equal(f.verdict, "partly");
    assert.match(f.nextStep, /BCR::ABL1/);
  }],
  ["blast phase flagged at 20% blasts", () => assert.match(checkCML({ bcrAbl1: "positive", blastsPct: 25 }).reasons.join(" "), /blast phase/)],
  ["20% blasts is in the acute-leukemia range", () => assert.equal(checkBlasts({ blastsPct: 20 }).verdict, "meets")],
  ["12% blasts is raised but below 20%", () => assert.equal(checkBlasts({ blastsPct: 12 }).verdict, "partly")],
  ["no data means cannot assess", () => {
    assert.equal(checkCLL({}).verdict, "cannot_assess");
    assert.equal(checkCML({}).verdict, "cannot_assess");
    assert.equal(checkBlasts({}).verdict, "cannot_assess");
  }],
];

let failed = 0;
for (const [name, fn] of cases) {
  try {
    fn();
    console.log(`ok   ${name}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name}: ${(e as Error).message}`);
  }
}
console.log(`${cases.length - failed}/${cases.length} passed`);
process.exit(failed ? 1 : 0);
