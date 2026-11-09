import Link from "next/link";

export function SiteFooter() {
  return (
    <div className="footer-outer">
      <footer className="footer">
        <div className="footer-cols">
          <div>
            <span className="label">Explore</span>
            <ul>
              <li><Link href="/">Analyse a cell</Link></li>
              <li><Link href="/model-card">Model card</Link></li>
              <li><Link href="/method">Method</Link></li>
            </ul>
          </div>
          <div>
            <span className="label">Built by</span>
            <ul>
              <li><span>Arnav Srivastava</span></li>
              <li><span>HaiCLOP Labs</span></li>
              <li><span>Dibrugarh, Assam</span></li>
            </ul>
            <p className="fine">Independent research and open source. Results are validation numbers until the held-out test run is published.</p>
          </div>
          <div>
            <span className="label">Data</span>
            <ul>
              <li><a href="https://doi.org/10.7937/tcia.2019.dc64i46r" rel="noopener noreferrer">C-NMC 2019 on TCIA</a></li>
              <li><span>CC BY 3.0</span></li>
            </ul>
            <p className="fine">Research demo. Not for clinical use. Uploaded images are not stored unless you opt in, and then for 24 hours only.</p>
          </div>
        </div>
        <p className="wordmark" aria-hidden="true">cell triage</p>
      </footer>
    </div>
  );
}
