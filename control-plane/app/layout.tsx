import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TokenCircuit — Control Plane",
  description: "Monitor and configure TokenCircuit loop detection",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-tc-bg text-tc-text antialiased">
        <nav className="border-b border-tc-border bg-tc-card px-6 py-3">
          <div className="mx-auto flex max-w-6xl items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold text-tc-accent">TC</span>
              <span className="text-sm font-medium text-tc-muted">
                TokenCircuit
              </span>
            </div>
            <div className="flex items-center gap-6 text-sm">
              <a
                href="/dashboard"
                className="text-tc-muted transition-colors hover:text-tc-text"
              >
                Dashboard
              </a>
              <a
                href="/config"
                className="text-tc-muted transition-colors hover:text-tc-text"
              >
                Config
              </a>
            </div>
          </div>
        </nav>
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
