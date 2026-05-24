/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        danger: "#ef4444",
        warning: "#f97316",
        safe: "#22c55e",
        dark: "#0f172a",
        card: "#1e293b",
        border: "#334155",
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "monospace"],
        sans: ["'DM Sans'", "sans-serif"],
      },
    },
  },
  plugins: [],
};
