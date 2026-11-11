import Link from "next/link";

const REPO = "https://github.com/HaiCLOP/luekemia_cell_model";

export function SiteFooter() {
  return (
    <footer className="footer">
      <div className="wrap footer-inner">
        <div>
          <p className="big">
            A second opinion
            <br />
            <em>that knows its limits.</em>
          </p>
          <p className="fine">Research demo. Not for clinical use. Uploaded images are not stored unless you opt in, and then for 24 hours only.</p>
        </div>
        <div>
          <span className="label">Explore</span>
          <ul>
            <li><Link href="/">Analyse a cell</Link></li>
            <li><Link href="/smear">Analyse a smear</Link></li>
            <li><Link href="/blood-count">Check a blood count (CLL, CML)</Link></li>
            <li><Link href="/model-card">Model card</Link></li>
            <li><Link href="/method">Method</Link></li>
            <li><a href={REPO} rel="noopener noreferrer">Source code</a></li>
          </ul>
        </div>
        <div>
          <span className="label">Data</span>
          <ul>
            <li><a href="https://doi.org/10.7937/tcia.2019.dc64i46r" rel="noopener noreferrer">C-NMC 2019, TCIA</a></li>
            <li><span>CC BY 3.0</span></li>
            <li><a href="https://github.com/Shenggan/BCCD_Dataset" rel="noopener noreferrer">BCCD sample smears, MIT</a></li>
          </ul>
          <p className="fine">Code MIT. Weights CC BY 4.0. Credit appreciated.</p>
        </div>
      </div>
      <div className="wrap footer-base">
        <span>Built by Arnav Srivastava · HaiCLOP Labs · Dibrugarh, Assam</span>
        <span>Specimen CT-01</span>
      </div>
    </footer>
  );
}
