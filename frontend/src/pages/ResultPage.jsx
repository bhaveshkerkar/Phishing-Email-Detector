import React from "react";

const cardStyle = {
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: "16px",
  padding: "24px",
  marginBottom: "20px",
};

export default function ResultPage({ result, goTo }) {
  if (!result) {
    return (
      <div style={{ textAlign: "center", marginTop: "80px" }}>
        <p style={{ color: "#64748b", fontSize: "1rem" }}>
          No result yet. Go to the Analyzer first.
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
          }}
        >
          Go to Analyzer
        </button>
      </div>
    );
  }

  const isPhishing = result.prediction === 1;
  const conf = result.confidence;

  const verdictColor = isPhishing ? "#ef4444" : "#22c55e";
  const verdictBg = isPhishing ? "#450a0a" : "#052e16";
  const verdictBorder = isPhishing ? "#7f1d1d" : "#14532d";

  // Confidence bar color
  const barColor = conf >= 80 ? "#ef4444" : conf >= 50 ? "#f97316" : "#22c55e";

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: "28px", textAlign: "center" }}>
        <h2
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "1.6rem",
            fontWeight: 700,
            color: "#f1f5f9",
            marginBottom: "6px",
          }}
        >
          Analysis Report
        </h2>
        <p style={{ color: "#64748b", fontSize: "0.88rem" }}>
          {result.timestamp} &nbsp;|&nbsp; {result.sender || "Unknown sender"}
        </p>
      </div>

      {/* Verdict banner */}
      <div
        style={{
          ...cardStyle,
          background: verdictBg,
          border: `1px solid ${verdictBorder}`,
          textAlign: "center",
          marginBottom: "20px",
        }}
      >
        <div style={{ fontSize: "3rem", marginBottom: "8px" }}>
          {isPhishing ? "🚨" : "✅"}
        </div>
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "2rem",
            fontWeight: 700,
            color: verdictColor,
          }}
        >
          {result.label}
        </div>
        <div style={{ color: "#94a3b8", fontSize: "0.9rem", marginTop: "6px" }}>
          Risk Level:{" "}
          <span style={{ color: verdictColor, fontWeight: 700 }}>
            {result.risk_level}
          </span>
        </div>
        {result.subject && (
          <div
            style={{
              marginTop: "12px",
              color: "#64748b",
              fontSize: "0.85rem",
              fontStyle: "italic",
            }}
          >
            "{result.subject}"
          </div>
        )}
      </div>

      {/* Confidence meter */}
      <div style={cardStyle}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: "10px",
          }}
        >
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "0.8rem",
              color: "#94a3b8",
              fontWeight: 600,
            }}
          >
            PHISHING CONFIDENCE
          </span>
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "1rem",
              fontWeight: 700,
              color: barColor,
            }}
          >
            {conf}%
          </span>
        </div>

        {/* Bar track */}
        <div
          style={{
            background: "#0f172a",
            borderRadius: "999px",
            height: "12px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${conf}%`,
              height: "100%",
              background: barColor,
              borderRadius: "999px",
              transition: "width 0.8s ease",
            }}
          />
        </div>

        {/* Scale labels */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "6px",
            fontSize: "0.75rem",
            color: "#475569",
          }}
        >
          <span>Safe (0%)</span>
          <span>Medium (50%)</span>
          <span>High Risk (100%)</span>
        </div>
      </div>

      {/* Red flags */}
      <div style={cardStyle}>
        <h3
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "0.85rem",
            fontWeight: 700,
            color: "#94a3b8",
            marginBottom: "14px",
            letterSpacing: "0.05em",
          }}
        >
          🚩 RED FLAGS DETECTED ({result.red_flags?.length || 0})
        </h3>

        {result.red_flags?.length === 0 ? (
          <p style={{ color: "#22c55e", fontSize: "0.9rem" }}>
            ✅ No red flags found — email looks clean.
          </p>
        ) : (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {result.red_flags.map((flag, i) => (
              <li
                key={i}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: "10px",
                  padding: "10px 0",
                  borderBottom:
                    i < result.red_flags.length - 1
                      ? "1px solid #1e293b"
                      : "none",
                }}
              >
                <span style={{ color: "#ef4444", marginTop: "2px" }}>▸</span>
                <span style={{ color: "#fca5a5", fontSize: "0.9rem" }}>
                  {flag}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Feature breakdown */}
      <div style={cardStyle}>
        <h3
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "0.85rem",
            fontWeight: 700,
            color: "#94a3b8",
            marginBottom: "14px",
            letterSpacing: "0.05em",
          }}
        >
          📊 FEATURE BREAKDOWN
        </h3>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "10px",
          }}
        >
          {result.features &&
            Object.entries(result.features).map(([key, val]) => {
              const isHigh = val >= 1;
              return (
                <div
                  key={key}
                  style={{
                    background: "#0f172a",
                    borderRadius: "8px",
                    padding: "10px 12px",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <span
                    style={{
                      color: "#64748b",
                      fontSize: "0.78rem",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    {key.replace(/_/g, " ")}
                  </span>
                  <span
                    style={{
                      color: isHigh ? "#f97316" : "#22c55e",
                      fontWeight: 700,
                      fontSize: "0.85rem",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    {val}
                  </span>
                </div>
              );
            })}
        </div>
      </div>

      {/* Action buttons */}
      <div style={{ display: "flex", gap: "12px", marginTop: "8px" }}>
        <button
          onClick={() => goTo("analyzer")}
          style={{
            flex: 1,
            background: "#0ea5e9",
            color: "#fff",
            border: "none",
            borderRadius: "10px",
            padding: "13px",
            fontSize: "0.95rem",
            fontWeight: 700,
            cursor: "pointer",
            fontFamily: "'DM Sans', sans-serif",
          }}
        >
          🔍 Analyze Another Email
        </button>
        <button
          onClick={() => goTo("history")}
          style={{
            background: "transparent",
            color: "#94a3b8",
            border: "1px solid #334155",
            borderRadius: "10px",
            padding: "13px 20px",
            fontSize: "0.9rem",
            cursor: "pointer",
            fontFamily: "'DM Sans', sans-serif",
          }}
        >
          🕒 View History
        </button>
      </div>
    </div>
  );
}
