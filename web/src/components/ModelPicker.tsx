"use client";

import { useId } from "react";
import { RUN_IN_BROWSER } from "@/lib/api";
import { MODELS, type ModelName } from "@/lib/models";

/** Choose which model reads the cell(s). MH51401102 is the default. */
export function ModelPicker({ value, onChange, disabled }: { value: ModelName; onChange: (m: ModelName) => void; disabled?: boolean }) {
  const group = useId();
  return (
    <fieldset className="model-pick" disabled={disabled}>
      <legend className="small">Which model reads the cells</legend>
      <div className="model-options">
        {Object.values(MODELS).map((m) => (
          <label key={m.name} className={`model-option${value === m.name ? " on" : ""}`}>
            <input type="radio" name={group} value={m.name} checked={value === m.name} onChange={() => onChange(m.name)} />
            <span className="model-name">{m.label}</span>
            <span className="small">{m.what}</span>
            {RUN_IN_BROWSER && <span className="label">{m.mb} MB download, first use only</span>}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
