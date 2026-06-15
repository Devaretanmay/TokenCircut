/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        tc: {
          bg: "#0f1117",
          card: "#1a1d2e",
          border: "#2a2d3e",
          accent: "#7c3aed",
          "accent-hover": "#6d28d9",
          text: "#e2e8f0",
          muted: "#94a3b8",
          success: "#22c55e",
          warning: "#eab308",
          danger: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};
