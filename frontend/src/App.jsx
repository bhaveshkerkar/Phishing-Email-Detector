import React, { useState } from "react";
import Navbar from "./components/Navbar";
import AnalyzerPage from "./pages/AnalyzerPage";
import ResultPage from "./pages/ResultPage";
import HistoryPage from "./pages/HistoryPage";

export default function App() {
  const [currentPage, setCurrentPage] = useState("analyzer");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  // Called by AnalyzerPage when scan completes
  const handleResult = (data, emailInput) => {
    const entry = {
      ...data,
      subject: emailInput.subject,
      sender: emailInput.sender,
      timestamp: new Date().toLocaleString(),
      id: Date.now(),
    };
    setResult(entry);
    setHistory((prev) => [entry, ...prev].slice(0, 20)); // keep last 20
    setCurrentPage("result");
  };

  const goTo = (page) => setCurrentPage(page);

  return (
    <div className="min-h-screen" style={{ background: "#0f172a" }}>
      <Navbar currentPage={currentPage} goTo={goTo} />

      <main className="max-w-4xl mx-auto px-4 py-8">
        {currentPage === "analyzer" && <AnalyzerPage onResult={handleResult} />}
        {currentPage === "result" && <ResultPage result={result} goTo={goTo} />}
        {currentPage === "history" && (
          <HistoryPage history={history} goTo={goTo} setResult={setResult} />
        )}
      </main>
    </div>
  );
}
