// Web worker: runs the Python smear cell finder (src/vision/segment.py) with Pyodide, off the page's main thread.
// Pyodide and its packages come from the jsDelivr CDN at a pinned version; the browser caches them after first use.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
importScripts(PYODIDE + "pyodide.js");

let ready = null;

async function boot() {
  postMessage({ type: "status", text: "Loading Python and its image libraries (about 50 MB, only the first time)" });
  const py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage(["numpy", "scipy", "scikit-image", "opencv-python", "pillow"]);
  for (const f of ["segment.py", "smear_glue.py"]) {
    const src = await (await fetch(`/py/${f}`)).text();
    py.FS.writeFile(`/home/pyodide/${f}`, src);
  }
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide'); import smear_glue");
  return py;
}

function bytesOf(proxy) {
  const buf = proxy.getBuffer("u8");
  try {
    return { data: new Uint8Array(buf.data), shape: buf.shape };
  } finally {
    buf.release();
  }
}

self.onmessage = async (e) => {
  const { id, file, umPerPx, stainJson, modelSize, filterSize } = e.data;
  try {
    if (!ready) ready = boot();
    const py = await ready;
    postMessage({ type: "status", text: "Finding white cells in the smear" });
    const data = new Uint8Array(await file.arrayBuffer());
    const analyse = py.globals.get("smear_glue").analyse;
    const r = analyse(data, umPerPx || 0, stainJson, modelSize, filterSize);
    const get = (k) => r.get(k);
    const toJs = (v) => (v && v.toJs ? v.toJs() : v);
    const out = {
      image: bytesOf(get("image")),
      crops: bytesOf(get("crops")),
      filterCrops: bytesOf(get("filter_crops")),
      width: get("width"), height: get("height"), shrink: get("shrink"), source: get("source"),
      rbc: get("rbc_diameter_px"), scale: get("scale"), notes: toJs(get("notes")),
      cells: JSON.parse(get("cells_json")), scored: toJs(get("scored")),
    };
    r.destroy();
    postMessage({ type: "result", id, out }, [out.image.data.buffer, out.crops.data.buffer, out.filterCrops.data.buffer]);
  } catch (err) {
    postMessage({ type: "error", id, message: String(err && err.message ? err.message : err).slice(-600) });
  }
};
