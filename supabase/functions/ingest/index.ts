import { serve } from "https://deno.land/std@0.208.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";

interface TelemetryPayload {
  agency_id: string;
  client_id: string;
  agent_framework: string;
  signal_type: string;
  node_name: string;
  iterations_at_detection: number;
  model_name: string;
  estimated_tokens_saved: number;
  estimated_cost_saved_usd: number;
  timestamp?: string;
}

const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

function validateApiKey(apiKey: string): boolean {
  return apiKey.length > 0;
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "apikey, Authorization, Content-Type",
      },
    });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  const apiKey = req.headers.get("apikey");
  if (!apiKey || !validateApiKey(apiKey)) {
    return new Response(
      JSON.stringify({ error: "Unauthorized: invalid or missing apikey" }),
      { status: 401, headers: { "Content-Type": "application/json" } },
    );
  }

  try {
    const payload: TelemetryPayload = await req.json();

    if (
      !payload.agency_id || !payload.client_id || !payload.signal_type ||
      !payload.node_name
    ) {
      return new Response(
        JSON.stringify({
          error: "Bad request: missing required fields",
          required: [
            "agency_id",
            "client_id",
            "signal_type",
            "node_name",
          ],
        }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    const supabase = createClient(supabaseUrl, supabaseKey);

    const { data: agency, error: agencyError } = await supabase
      .from("agency_configs")
      .select("agency_id")
      .eq("agency_id", payload.agency_id)
      .single();

    if (agencyError || !agency) {
      return new Response(
        JSON.stringify({ error: "Unauthorized: unknown agency_id" }),
        { status: 401, headers: { "Content-Type": "application/json" } },
      );
    }

    const { error: insertError } = await supabase
      .from("telemetry_events")
      .insert({
        agency_id: payload.agency_id,
        client_id: payload.client_id,
        agent_framework: payload.agent_framework,
        signal_type: payload.signal_type,
        node_name: payload.node_name,
        iterations_at_detection: payload.iterations_at_detection,
        model_name: payload.model_name,
        estimated_tokens_saved: payload.estimated_tokens_saved,
        estimated_cost_saved_usd: payload.estimated_cost_saved_usd,
        created_at: payload.timestamp ?? new Date().toISOString(),
      });

    if (insertError) {
      console.error("Insert error:", insertError);
      return new Response(
        JSON.stringify({ error: "Failed to insert telemetry event" }),
        { status: 500, headers: { "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({ ok: true }),
      {
        status: 201,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      },
    );
  } catch (err) {
    console.error("Ingest error:", err);
    return new Response(
      JSON.stringify({ error: "Internal server error" }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
});
