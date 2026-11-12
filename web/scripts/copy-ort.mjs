// Copy runtime files into public/ so the site serves them itself:
//  - ONNX Runtime Web's WebAssembly files (public/ort)
//  - the Python smear cell finder, src/vision/segment.py (public/py), run in the browser by Pyodide.
//    The copy is committed because Vercel builds only the web/ folder; tests check it matches the source.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const ort = join(root, "public", "ort");
mkdirSync(ort, { recursive: true });
for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"])
  copyFileSync(join(root, "node_modules", "onnxruntime-web", "dist", f), join(ort, f));
const seg = join(root, "..", "src", "vision", "segment.py");
if (existsSync(seg)) copyFileSync(seg, join(root, "public", "py", "segment.py"));
console.log("copied ONNX Runtime Web files and the smear cell finder");
