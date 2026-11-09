import { createClient } from "@supabase/supabase-js";
import results from "@/data/results.json";

export interface ActiveModel {
  name: string;
  architecture: string;
  temperature: number;
  t_low: number;
  t_high: number;
  band_target: string;
  source: "database" | "bundled";
}

/** Reads the active model row with the public (anon) key; RLS allows only `is_active = true` rows. */
export async function getActiveModel(): Promise<ActiveModel> {
  const fallback: ActiveModel = { ...results.model, source: "bundled" };
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return fallback;
  try {
    const sb = createClient(url, key, { auth: { persistSession: false } });
    const { data, error } = await sb
      .from("model_versions")
      .select("name, architecture, temperature, t_low, t_high, band_target")
      .eq("is_active", true)
      .maybeSingle();
    if (error || !data) return fallback;
    return { ...data, source: "database" } as ActiveModel;
  } catch {
    return fallback;
  }
}
