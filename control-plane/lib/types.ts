export interface AgencyConfig {
  agency_id: string;
  client_id: string;
  max_repeats: number;
  window_size: number;
  webhook_url: string | null;
  updated_at: string;
}

export interface AgencyWeeklySummary {
  agency_id: string;
  client_id: string;
  loops_intercepted: number;
  total_saved_usd: number;
  last_event_at: string | null;
}
