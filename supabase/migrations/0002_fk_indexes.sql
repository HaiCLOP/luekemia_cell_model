-- Covering indexes for foreign keys (Supabase performance advisor 0001_unindexed_foreign_keys).
create index if not exists feedback_prediction_id_idx on feedback (prediction_id);
create index if not exists predictions_model_version_id_idx on predictions (model_version_id);
create index if not exists training_runs_model_version_id_idx on training_runs (model_version_id);

-- Note: tables other than model_versions intentionally have RLS enabled with no policies, so only the
-- service role (server side) can read or write them. Advisor 0008 reports this as INFO; it is by design.
