"use client";

import { useState } from "react";
import { createBrowser } from "@/lib/supabase/browser";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const supabase = createBrowser();
    const { error: signInError } = await supabase.auth.signInWithOtp({
      email,
      options: {
        shouldCreateUser: true,
      },
    });
    if (signInError) {
      setError(signInError.message);
    } else {
      setSent(true);
    }
  }

  if (sent) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="w-full max-w-md rounded-lg border border-tc-border bg-tc-card p-8 text-center">
          <h1 className="mb-2 text-xl font-semibold">Check your email</h1>
          <p className="text-tc-muted">
            A magic sign-in link has been sent to <strong>{email}</strong>.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="w-full max-w-md rounded-lg border border-tc-border bg-tc-card p-8">
        <h1 className="mb-6 text-2xl font-semibold">Sign in to TokenCircuit</h1>
        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label
              htmlFor="email"
              className="mb-1 block text-sm font-medium text-tc-muted"
            >
              Email address
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full rounded-md border border-tc-border bg-tc-bg px-3 py-2 text-sm text-tc-text placeholder-tc-muted focus:border-tc-accent focus:outline-none"
              placeholder="you@example.com"
            />
          </div>
          {error && (
            <p className="text-sm text-tc-danger">{error}</p>
          )}
          <button
            type="submit"
            className="w-full rounded-md bg-tc-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-tc-accent-hover"
          >
            Send magic link
          </button>
        </form>
      </div>
    </div>
  );
}
