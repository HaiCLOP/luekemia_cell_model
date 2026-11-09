import Link from "next/link";

export default function NotFound() {
  return (
    <section className="prose section center">
      <h1 className="display" style={{ fontSize: "clamp(40px,6vw,64px)" }}>
        Nothing here.
        <br />
        <em>Try the analyser.</em>
      </h1>
      <div className="actions" style={{ justifyContent: "center" }}>
        <Link href="/" className="btn btn-primary">Go home</Link>
      </div>
    </section>
  );
}
