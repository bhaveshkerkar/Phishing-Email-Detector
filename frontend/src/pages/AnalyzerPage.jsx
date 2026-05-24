import React, { useState } from "react";
import { analyzeEmail } from "../utils/api";

const inputStyle = {
  width: "100%",
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: "10px",
  color: "#f1f5f9",
  padding: "12px 14px",
  fontFamily: "'DM Sans', sans-serif",
  fontSize: "0.92rem",
  outline: "none",
  transition: "border 0.2s",
};

const labelStyle = {
  display: "block",
  marginBottom: "6px",
  fontSize: "0.82rem",
  fontWeight: 600,
  color: "#94a3b8",
  fontFamily: "'JetBrains Mono', monospace",
  letterSpacing: "0.05em",
};

export default function AnalyzerPage({ onResult }) {
  const [form, setForm] = useState({
    subject: "",
    sender: "",
    body: "",
    url: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError("");
  };

  const handleSubmit = async () => {
    if (!form.subject && !form.body) {
      setError("Please enter at least a subject or email body.");
      return;
    }
    setLoading(true);
    setError("");
    const { success, data, error: err } = await analyzeEmail(form);
    setLoading(false);
    if (!success) {
      setError(err);
      return;
    }
    onResult(data, form);
  };

  const handleClear = () => {
    setForm({ subject: "", sender: "", body: "", url: "" });
    setError("");
  };

  const loadSample = (type) => {
    if (type === "phishing") {
      setForm({
        subject: "URGENT: Your account has been compromised!",
        sender: "security@paypa1-support.com",
        body: "DEAR USER,\n\nYour account has been SUSPENDED due to suspicious activity!! Click the link below IMMEDIATELY to verify your identity or your account will be PERMANENTLY DELETED within 24 hours.\n\nClick here: http://paypa1-secure-login.xyz/verify?token=abc123\n\nDo NOT ignore this message.",
        url: "http://paypa1-secure-login.xyz/verify?token=abc123",
      });
    } else {
      setForm({
        subject: "Your monthly account statement is ready",
        sender: "billing@stripe.com",
        body: "Hi there,\n\nYour monthly statement for June 2026 is now available in your account dashboard. Please log in to view it at your convenience.\n\nBest regards,\nThe Billing Team",
        url: "https://stripe.com/billing",
      });
    }
    setError("");
  };

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: "32px", textAlign: "center" }}>
        <h1
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "2rem",
            fontWeight: 700,
            color: "#f1f5f9",
            marginBottom: "8px",
          }}
        >
          🛡️ Phishing Email Detector
        </h1>
        <p style={{ color: "#64748b", fontSize: "0.95rem" }}>
          Paste email details below — AI will analyze it for phishing signals
        </p>
      </div>

      {/* Sample buttons */}
      <div
        style={{
          display: "flex",
          gap: "10px",
          marginBottom: "24px",
          justifyContent: "center",
        }}
      >
        <span
          style={{ color: "#64748b", fontSize: "0.85rem", alignSelf: "center" }}
        >
          Try a sample:
        </span>
        <button
          onClick={() => loadSample("phishing")}
          style={{
            background: "#450a0a",
            color: "#fca5a5",
            border: "1px solid #7f1d1d",
            borderRadius: "8px",
            padding: "6px 14px",
            cursor: "pointer",
            fontSize: "0.83rem",
            fontWeight: 600,
          }}
        >
          ⚠️ Phishing Sample
        </button>
        <button
          onClick={() => loadSample("legit")}
          style={{
            background: "#052e16",
            color: "#86efac",
            border: "1px solid #14532d",
            borderRadius: "8px",
            padding: "6px 14px",
            cursor: "pointer",
            fontSize: "0.83rem",
            fontWeight: 600,
          }}
        >
          ✅ Legit Sample
        </button>
      </div>

      {/* Form card */}
      <div
        style={{
          background: "#1e293b",
          border: "1px solid #334155",
          borderRadius: "16px",
          padding: "28px",
        }}
      >
        {/* Subject */}
        <div style={{ marginBottom: "18px" }}>
          <label style={labelStyle}>SUBJECT</label>
          <input
            style={inputStyle}
            type="text"
            name="subject"
            placeholder="e.g. URGENT: Your account has been suspended"
            value={form.subject}
            onChange={handleChange}
            onFocus={(e) => (e.target.style.border = "1px solid #0ea5e9")}
            onBlur={(e) => (e.target.style.border = "1px solid #334155")}
          />
        </div>

        {/* Sender */}
        <div style={{ marginBottom: "18px" }}>
          <label style={labelStyle}>SENDER EMAIL</label>
          <input
            style={inputStyle}
            type="text"
            name="sender"
            placeholder="e.g. security@paypa1-support.com"
            value={form.sender}
            onChange={handleChange}
            onFocus={(e) => (e.target.style.border = "1px solid #0ea5e9")}
            onBlur={(e) => (e.target.style.border = "1px solid #334155")}
          />
        </div>

        {/* Body */}
        <div style={{ marginBottom: "18px" }}>
          <label style={labelStyle}>EMAIL BODY</label>
          <textarea
            style={{ ...inputStyle, minHeight: "160px", resize: "vertical" }}
            name="body"
            placeholder="Paste the full email body here..."
            value={form.body}
            onChange={handleChange}
            onFocus={(e) => (e.target.style.border = "1px solid #0ea5e9")}
            onBlur={(e) => (e.target.style.border = "1px solid #334155")}
          />
        </div>

        {/* URL */}
        <div style={{ marginBottom: "24px" }}>
          <label style={labelStyle}>SUSPICIOUS URL (optional)</label>
          <input
            style={inputStyle}
            type="text"
            name="url"
            placeholder="e.g. http://paypa1-secure-login.xyz/verify"
            value={form.url}
            onChange={handleChange}
            onFocus={(e) => (e.target.style.border = "1px solid #0ea5e9")}
            onBlur={(e) => (e.target.style.border = "1px solid #334155")}
          />
        </div>

        {/* Error */}
        {error && (
          <div
            style={{
              background: "#450a0a",
              border: "1px solid #7f1d1d",
              borderRadius: "8px",
              padding: "10px 14px",
              color: "#fca5a5",
              fontSize: "0.87rem",
              marginBottom: "16px",
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* Buttons */}
        <div style={{ display: "flex", gap: "12px" }}>
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{
              flex: 1,
              background: loading ? "#1e40af" : "#0ea5e9",
              color: "#fff",
              border: "none",
              borderRadius: "10px",
              padding: "13px",
              fontSize: "1rem",
              fontWeight: 700,
              cursor: loading ? "not-allowed" : "pointer",
              fontFamily: "'DM Sans', sans-serif",
              transition: "background 0.2s",
            }}
          >
            {loading ? "🔄 Analyzing..." : "🔍 Analyze Email"}
          </button>
          <button
            onClick={handleClear}
            style={{
              background: "transparent",
              color: "#64748b",
              border: "1px solid #334155",
              borderRadius: "10px",
              padding: "13px 20px",
              fontSize: "0.9rem",
              cursor: "pointer",
              fontFamily: "'DM Sans', sans-serif",
            }}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
