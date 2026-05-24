import React from "react";

export default function HistoryPage({ history, goTo, setResult }) {
  if (history.length === 0) {
    return (
      <div style={{ textAlign: "center", marginTop: "80px" }}>
        <p style={{ color: "#64748b", fontSize: "1rem" }}>
          No scans yet. Analyze an email first.
        </p>
        <button
          onClick={() => goTo("analyzer")}
          style={{
            marginTop: "16px",
            background: "#0ea5e9",
            color: "#fff",
            border: "none",
            borderRadius: "10px",
            padding: "10px 24px",
            cursor: "pointer",
            fontWeight: 700,
            fontFamily: "'DM Sans', sans-serif",
          }}
        >
          Go to Analyzer
        </button>
      </div>
    );
  }

  const handleView = (entry) => {
    setResult(entry);
    goTo("result");
  };

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: "28px" }}>
        <h2
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "1.6rem",
            fontWeight: 700,
            color: "#f1f5f9",
            marginBottom: "6px",
          }}
        >
          🕒 Scan History
        </h2>
        <p style={{ color: "#64748b", fontSize: "0.88rem" }}>
          {history.length} scan{history.length !== 1 ? "s" : ""} recorded this
          session
        </p>
      </div>

      {/* History list */}
      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {history.map((entry, i) => {
          const isPhishing = entry.prediction === 1;
          const verdictColor = isPhishing ? "#ef4444" : "#22c55e";
          const verdictBg = isPhishing ? "#450a0a" : "#052e16";
          const verdictBorder = isPhishing ? "#7f1d1d" : "#14532d";
          const barColor =
            entry.confidence >= 80
              ? "#ef4444"
              : entry.confidence >= 50
                ? "#f97316"
                : "#22c55e";

          return (
            <div
              key={entry.id}
              style={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "14px",
                padding: "18px 20px",
                display: "flex",
                alignItems: "center",
                gap: "16px",
                cursor: "pointer",
                transition: "border 0.2s",
              }}
              onClick={() => handleView(entry)}
              onMouseEnter={(e) =>
                (e.currentTarget.style.border = "1px solid #0ea5e9")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.border = "1px solid #334155")
              }
            >
              {/* Index */}
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "0.8rem",
                  color: "#475569",
                  minWidth: "24px",
                }}
              >
                #{history.length - i}
              </div>

              {/* Verdict badge */}
              <div
                style={{
                  background: verdictBg,
                  border: `1px solid ${verdictBorder}`,
                  borderRadius: "8px",
                  padding: "4px 10px",
                  minWidth: "100px",
                  textAlign: "center",
                }}
              >
                <span
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "0.75rem",
                    fontWeight: 700,
                    color: verdictColor,
                  }}
                >
                  {isPhishing ? "🚨 PHISHING" : "✅ LEGIT"}
                </span>
              </div>

              {/* Subject + sender */}
              <div style={{ flex: 1, overflow: "hidden" }}>
                <div
                  style={{
                    color: "#f1f5f9",
                    fontSize: "0.9rem",
                    fontWeight: 600,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {entry.subject || "(No subject)"}
                </div>
                <div
                  style={{
                    color: "#64748b",
                    fontSize: "0.78rem",
                    marginTop: "2px",
                  }}
                >
                  {entry.sender || "Unknown sender"} &nbsp;·&nbsp;{" "}
                  {entry.timestamp}
                </div>
              </div>

              {/* Confidence bar */}
              <div style={{ minWidth: "100px" }}>
                <div
                  style={{
                    fontSize: "0.72rem",
                    color: "#64748b",
                    marginBottom: "4px",
                    textAlign: "right",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  {entry.confidence}%
                </div>
                <div
                  style={{
                    background: "#0f172a",
                    borderRadius: "999px",
                    height: "6px",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${entry.confidence}%`,
                      height: "100%",
                      background: barColor,
                      borderRadius: "999px",
                    }}
                  />
                </div>
              </div>

              {/* Arrow */}
              <div style={{ color: "#334155", fontSize: "1rem" }}>›</div>
            </div>
          );
        })}
      </div>

      {/* Back button */}
      <button
        onClick={() => goTo("analyzer")}
        style={{
          marginTop: "24px",
          width: "100%",
          background: "transparent",
          color: "#64748b",
          border: "1px solid #334155",
          borderRadius: "10px",
          padding: "12px",
          fontSize: "0.9rem",
          cursor: "pointer",
          fontFamily: "'DM Sans', sans-serif",
        }}
      >
        + Analyze New Email
      </button>
    </div>
  );
}
