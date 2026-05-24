import React from "react";

export default function Navbar({ currentPage, goTo }) {
  const links = [
    { id: "analyzer", label: "🔍 Analyze" },
    { id: "result", label: "📊 Result" },
    { id: "history", label: "🕒 History" },
  ];

  return (
    <nav
      style={{
        background: "#1e293b",
        borderBottom: "1px solid #334155",
      }}
    >
      <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <span style={{ fontSize: "1.5rem" }}>🛡️</span>
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: 700,
              fontSize: "1.1rem",
              color: "#38bdf8",
            }}
          >
            PhishGuard
          </span>
          <span
            style={{
              fontSize: "0.65rem",
              background: "#0ea5e9",
              color: "#fff",
              padding: "2px 7px",
              borderRadius: "999px",
              fontWeight: 600,
              marginLeft: "4px",
            }}
          >
            AI
          </span>
        </div>

        {/* Nav links */}
        <div className="flex gap-2">
          {links.map((link) => (
            <button
              key={link.id}
              onClick={() => goTo(link.id)}
              style={{
                padding: "6px 14px",
                borderRadius: "8px",
                border: "none",
                cursor: "pointer",
                fontFamily: "'DM Sans', sans-serif",
                fontWeight: 500,
                fontSize: "0.85rem",
                transition: "all 0.2s",
                background: currentPage === link.id ? "#0ea5e9" : "transparent",
                color: currentPage === link.id ? "#fff" : "#94a3b8",
              }}
            >
              {link.label}
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
}
