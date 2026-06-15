-- Agency configuration (pulled on SDK boot)
CREATE TABLE IF NOT EXISTS agency_configs (
  agency_id UUID PRIMARY KEY,
  client_id TEXT NOT NULL,
  max_repeats INT DEFAULT 5 CHECK (max_repeats >= 1),
  window_size INT DEFAULT 5 CHECK (window_size >= 1),
  webhook_url TEXT,
  updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE agency_configs ENABLE ROW LEVEL SECURITY;

-- Allow read access for authenticated users
CREATE POLICY "Users can read own configs"
  ON agency_configs
  FOR SELECT
  USING (auth.role() = 'authenticated');

-- Allow upsert for authenticated users
CREATE POLICY "Users can upsert own configs"
  ON agency_configs
  FOR INSERT
  WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Users can update own configs"
  ON agency_configs
  FOR UPDATE
  USING (auth.role() = 'authenticated');

-- Anonymized telemetry events
CREATE TABLE IF NOT EXISTS telemetry_events (
  id BIGSERIAL PRIMARY KEY,
  agency_id UUID REFERENCES agency_configs(agency_id),
  client_id TEXT,
  agent_framework TEXT,
  signal_type TEXT,
  node_name TEXT,
  iterations_at_detection INT,
  model_name TEXT,
  estimated_tokens_saved INT,
  estimated_cost_saved_usd FLOAT,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE telemetry_events ENABLE ROW LEVEL SECURITY;

-- Allow insert via Edge Function (service role)
CREATE POLICY "Service role can insert telemetry"
  ON telemetry_events
  FOR INSERT
  WITH CHECK (auth.role() = 'service_role');

-- Dashboard aggregate view
CREATE OR REPLACE VIEW agency_weekly_summary AS
SELECT
  agency_id,
  client_id,
  COUNT(*) as loops_intercepted,
  SUM(estimated_cost_saved_usd) as total_saved_usd,
  MAX(created_at) as last_event_at
FROM telemetry_events
WHERE created_at > now() - interval '7 days'
GROUP BY agency_id, client_id;

-- Health check function for keep-alive
CREATE OR REPLACE FUNCTION health_check()
RETURNS json
LANGUAGE sql
AS $$
  SELECT json_build_object('status', 'ok');
$$;
