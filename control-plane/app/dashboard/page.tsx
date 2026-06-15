import { createClient } from "@/lib/supabase/client";
import type { AgencyWeeklySummary } from "@/lib/types";

async function getSummary(): Promise<AgencyWeeklySummary | null> {
  const supabase = createClient();

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  const { data: configs } = await supabase
    .from("agency_configs")
    .select("agency_id, client_id")
    .eq("client_id", user.email)
    .limit(1);

  if (!configs || configs.length === 0) return null;

  const { data } = await supabase
    .from("agency_weekly_summary")
    .select("*")
    .eq("agency_id", configs[0].agency_id)
    .single();

  return data as AgencyWeeklySummary | null;
}

async function getFleetSize(): Promise<number> {
  const supabase = createClient();
  const { count } = await supabase
    .from("agency_configs")
    .select("*", { count: "exact", head: true });
  return count ?? 0;
}

export default async function DashboardPage() {
  const [summary, fleetSize] = await Promise.all([
    getSummary(),
    getFleetSize(),
  ]);

  const loopsIntercepted = summary?.loops_intercepted ?? 0;
  const marginSaved = summary?.total_saved_usd ?? 0;

  return (
    <div>
      <h1 className="mb-6 text-2xl font-semibold">Dashboard</h1>
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <div className="rounded-lg border border-tc-border bg-tc-card p-6">
          <p className="text-sm font-medium text-tc-muted">Fleet Size</p>
          <p className="mt-2 text-3xl font-bold">{fleetSize}</p>
          <p className="mt-1 text-xs text-tc-muted">Active agencies</p>
        </div>

        <div className="rounded-lg border border-tc-border bg-tc-card p-6">
          <p className="text-sm font-medium text-tc-muted">
            Loops Intercepted
          </p>
          <p className="mt-2 text-3xl font-bold text-tc-success">
            {loopsIntercepted}
          </p>
          <p className="mt-1 text-xs text-tc-muted">Past 7 days</p>
        </div>

        <div className="rounded-lg border border-tc-border bg-tc-card p-6">
          <p className="text-sm font-medium text-tc-muted">
            Estimated Margin Saved
          </p>
          <p className="mt-2 text-3xl font-bold text-tc-accent">
            ${marginSaved.toFixed(2)}
          </p>
          <p className="mt-1 text-xs text-tc-muted">(estimated) Past 7 days</p>
        </div>
      </div>

      <div className="mt-8 rounded-lg border border-tc-border bg-tc-card p-6">
        <h2 className="mb-2 text-sm font-medium text-tc-muted">
          About these metrics
        </h2>
        <p className="text-xs text-tc-muted">
          Loops Intercepted is the count of infinite loops detected and stopped
          by TokenCircuit. Estimated Margin Saved is computed using average
          model token costs per call and may differ from actual savings. All
          telemetry data is anonymized — no prompts, no argument values leave
          your infrastructure.
        </p>
      </div>
    </div>
  );
}
