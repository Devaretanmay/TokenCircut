"use client";

import { useEffect, useState } from "react";
import { createBrowser } from "@/lib/supabase/browser";
import type { AgencyConfig } from "@/lib/types";

export default function ConfigPage() {
  const [configs, setConfigs] = useState<AgencyConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);

  useEffect(() => {
    async function fetchConfigs() {
      const supabase = createBrowser();

      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (!user) {
        setLoading(false);
        return;
      }

      const { data } = await supabase
        .from("agency_configs")
        .select("*")
        .eq("client_id", user.email);

      setConfigs((data as AgencyConfig[]) ?? []);
      setLoading(false);
    }
    fetchConfigs();
  }, []);

  async function handleSave(config: AgencyConfig) {
    setSaving(config.agency_id);
    const supabase = createBrowser();
    const { error } = await supabase.from("agency_configs").upsert(
      {
        agency_id: config.agency_id,
        client_id: config.client_id,
        max_repeats: config.max_repeats,
        window_size: config.window_size,
        webhook_url: config.webhook_url,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "agency_id" }
    );
    setSaving(null);
    if (error) {
      alert("Failed to save: " + error.message);
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <p className="text-tc-muted">Loading...</p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="mb-6 text-2xl font-semibold">Configuration</h1>

      {configs.length === 0 && (
        <div className="rounded-lg border border-tc-border bg-tc-card p-8 text-center">
          <p className="text-tc-muted">
            No agency configurations found. Contact support to set up your
            agency.
          </p>
        </div>
      )}

      <div className="space-y-4">
        {configs.map((config) => (
          <ConfigCard
            key={config.agency_id}
            config={config}
            onSave={handleSave}
            saving={saving === config.agency_id}
          />
        ))}
      </div>
    </div>
  );
}

function ConfigCard({
  config,
  onSave,
  saving,
}: {
  config: AgencyConfig;
  onSave: (c: AgencyConfig) => void;
  saving: boolean;
}) {
  const [local, setLocal] = useState(config);

  return (
    <div className="rounded-lg border border-tc-border bg-tc-card p-6">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="font-semibold">{config.client_id}</h2>
          <p className="text-xs text-tc-muted">
            ID: {config.agency_id.slice(0, 8)}...
          </p>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs font-medium text-tc-muted">
            Max Repeats <span className="text-tc-muted">(3–10)</span>
          </label>
          <input
            type="number"
            min={3}
            max={10}
            value={local.max_repeats}
            onChange={(e) =>
              setLocal({ ...local, max_repeats: Number(e.target.value) })
            }
            className="w-full rounded-md border border-tc-border bg-tc-bg px-3 py-2 text-sm text-tc-text focus:border-tc-accent focus:outline-none"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-tc-muted">
            Window Size
          </label>
          <input
            type="number"
            min={2}
            max={20}
            value={local.window_size}
            onChange={(e) =>
              setLocal({ ...local, window_size: Number(e.target.value) })
            }
            className="w-full rounded-md border border-tc-border bg-tc-bg px-3 py-2 text-sm text-tc-text focus:border-tc-accent focus:outline-none"
          />
        </div>
      </div>

      <div className="mt-4">
        <label className="mb-1 block text-xs font-medium text-tc-muted">
          Webhook URL <span className="text-tc-muted">(optional)</span>
        </label>
        <input
          type="url"
          value={local.webhook_url ?? ""}
          onChange={(e) =>
            setLocal({ ...local, webhook_url: e.target.value || null })
          }
          placeholder="https://hooks.example.com/alert"
          className="w-full rounded-md border border-tc-border bg-tc-bg px-3 py-2 text-sm text-tc-text placeholder-tc-muted focus:border-tc-accent focus:outline-none"
        />
      </div>

      <div className="mt-4 flex justify-end">
        <button
          onClick={() => onSave(local)}
          disabled={saving}
          className="rounded-md bg-tc-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-tc-accent-hover disabled:opacity-50"
        >
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
    </div>
  );
}
