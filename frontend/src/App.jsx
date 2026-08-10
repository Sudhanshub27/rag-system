import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import ReadingPane from './components/ReadingPane';
import SourceInspector from './components/SourceInspector';

import HowItWorks from './pages/HowItWorks';
import Privacy from './pages/Privacy';
import RetrievalSettings from './pages/RetrievalSettings';
import FAQ from './pages/FAQ';

const API_BASE = 'http://localhost:8000/api';

function MainWorkspace() {
  const [tenantId, setTenantId] = useState('');
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState({});
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStreamText, setCurrentStreamText] = useState('');
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [fileProgress, setFileProgress] = useState({});
  const [error, setError] = useState(null);
  const [followups, setFollowups] = useState([]);

  // RAG Pipeline Settings & Features State
  const [settings, setSettings] = useState({
    splitView: true,
    debugScores: false,
    useHyde: false,
    useMultiQuery: false,
  });

  // Fetch Documents & Stats
  const refreshData = async () => {
    try {
      const [docsRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/documents`, { credentials: 'include' }),
        fetch(`${API_BASE}/stats`, { credentials: 'include' }),
      ]);
      if (docsRes.ok) {
        const d = await docsRes.json();
        setDocuments(d.documents || []);
        if (d.tenant_id) setTenantId(d.tenant_id);
      }
      if (statsRes.ok) {
        const s = await statsRes.json();
        setStats(s);
      }
    } catch (err) {
      console.error('Failed to load stats/docs:', err);
    }
  };

  useEffect(() => {
    refreshData();
  }, []);

  // Multi-stage SSE File Upload
  const handleUpload = async (files) => {
    setError(null);
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error(`Upload failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunkStr = decoder.decode(value);
        const lines = chunkStr.split('\n');

        let currentEvent = 'message';
        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEvent = line.replace('event:', '').trim();
          } else if (line.startsWith('data:')) {
            const dataStr = line.replace('data:', '').trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              setFileProgress((prev) => ({
                ...prev,
                [data.filename]: { ...data, event: currentEvent },
              }));
            } catch (e) {}
          }
        }
      }
      await refreshData();
    } catch (err) {
      setError(err.message);
    }
  };

  // SSE Answer Streaming Query with HyDE & Multi-Query Options
  const handleSendMessage = async (question) => {
    setError(null);
    setFollowups([]);
    const userMsg = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);
    setCurrentStreamText('');

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          use_hyde: settings.useHyde,
          use_multi_query: settings.useMultiQuery,
        }),
        credentials: 'include',
      });

      if (response.status === 429) {
        const errJson = await response.json();
        throw new Error(errJson.detail?.message || 'Rate limit exceeded');
      }

      if (!response.ok) {
        throw new Error(`Query failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let streamedText = '';
      let finalData = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunkStr = decoder.decode(value);
        const lines = chunkStr.split('\n');

        let currentEvent = 'token';
        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEvent = line.replace('event:', '').trim();
          } else if (line.startsWith('data:')) {
            const dataStr = line.replace('data:', '').trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              if (currentEvent === 'token') {
                streamedText += data.token;
                setCurrentStreamText(streamedText);
              } else if (currentEvent === 'final') {
                finalData = data;
              }
            } catch (e) {}
          }
        }
      }

      const assistantMsg = {
        role: 'assistant',
        content: streamedText,
        citations: finalData?.retrieved_chunks || [],
        faithfulness: finalData?.faithfulness_score,
        relevance: finalData?.relevance_score,
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Request contextual follow-up questions
      fetchFollowups(question, streamedText);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsStreaming(false);
      setCurrentStreamText('');
    }
  };

  const fetchFollowups = async (lastQ, lastA) => {
    try {
      const res = await fetch(`${API_BASE}/followups`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ last_question: lastQ, last_answer: lastA }),
      });
      if (res.ok) {
        const data = await res.json();
        setFollowups(data.followups || []);
      }
    } catch (e) {}
  };

  const handleSelectCheckpoint = (msgIndex) => {
    const el = document.getElementById(`checkpoint-${msgIndex}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleDeleteDoc = async (filename) => {
    try {
      await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
        credentials: 'include',
      });
      await refreshData();
    } catch (err) {
      setError(`Failed to delete document: ${err.message}`);
    }
  };

  const handleDeleteAllData = async () => {
    try {
      await fetch(`${API_BASE}/tenant/${tenantId}`, {
        method: 'DELETE',
        credentials: 'include',
      });
      setMessages([]);
      setSelectedCitation(null);
      await refreshData();
    } catch (err) {
      setError(`Failed to wipe data: ${err.message}`);
    }
  };

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <Sidebar
        tenantId={tenantId}
        documents={documents}
        stats={stats}
        messages={messages}
        onUpload={handleUpload}
        fileProgress={fileProgress}
        onDeleteDoc={handleDeleteDoc}
        onDeleteAllData={handleDeleteAllData}
        onSelectCheckpoint={handleSelectCheckpoint}
        settings={settings}
        onUpdateSettings={setSettings}
      />
      <ReadingPane
        messages={messages}
        onSendMessage={handleSendMessage}
        isStreaming={isStreaming}
        currentStreamText={currentStreamText}
        onSelectCitation={setSelectedCitation}
        error={error}
        followups={followups}
        onSelectFollowup={handleSendMessage}
        onSelectCheckpoint={handleSelectCheckpoint}
        debugScores={settings.debugScores}
      />
      {settings.splitView && (
        <SourceInspector
          selectedCitation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
          debugScores={settings.debugScores}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <div className="flex flex-col h-screen w-screen overflow-hidden bg-parchment-100 font-sans antialiased">
        <Navbar />
        <main className="flex-1 min-h-0 overflow-hidden relative flex">
          <Routes>
            <Route path="/" element={<MainWorkspace />} />
            <Route path="/how-it-works" element={<HowItWorks />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/retrieval-settings" element={<RetrievalSettings />} />
            <Route path="/faq" element={<FAQ />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
