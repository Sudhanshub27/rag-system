import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Sliders, Bookmark } from 'lucide-react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import ReadingPane from './components/ReadingPane';
import SourceInspector from './components/SourceInspector';

import HowItWorks from './pages/HowItWorks';
import Privacy from './pages/Privacy';
import RetrievalSettings from './pages/RetrievalSettings';
import FAQ from './pages/FAQ';

const getApiBase = () => {
  let base = import.meta.env.VITE_API_BASE_URL;
  if (base && base.trim() !== '') {
    base = base.trim().replace(/\/+$/, '');
    if (!base.endsWith('/api')) {
      base = `${base}/api`;
    }
    return base;
  }
  return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? '/api'
    : 'http://localhost:8000/api';
};

const API_BASE = getApiBase();

function MainWorkspace() {
  const [tenantId, setTenantId] = useState('');
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState({});
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('rag_chat_messages');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('rag_chat_messages', JSON.stringify(messages));
    } catch (e) {}
  }, [messages]);

  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStreamText, setCurrentStreamText] = useState('');
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [fileProgress, setFileProgress] = useState({});
  const [error, setError] = useState(null);
  const [followups, setFollowups] = useState([]);

  // RAG Pipeline Settings & Features State
  const [settings, setSettings] = useState(() => ({
    splitView: true,
    debugScores: false,
    useHyde: false,
    useMultiQuery: false,
    provider: localStorage.getItem('rag_provider') || 'groq',
    model: localStorage.getItem('rag_model') || '',
    apiKey: localStorage.getItem('rag_api_key') || '',
    anonymizePii: localStorage.getItem('rag_anonymize_pii') !== 'false',
  }));

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
      setTimeout(() => {
        setFileProgress({});
      }, 3500);
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
          provider: settings.provider,
          model: settings.model || null,
          api_key: settings.apiKey || null,
          anonymize_pii: true,
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
              } else if (currentEvent === 'error') {
                throw new Error(data.error || 'Failed to generate answer from provider');
              }
            } catch (e) {
              if (e.message && e.message.includes('Failed to generate')) {
                throw e;
              }
            }
          }
        }
      }

      const fallbackText = "I could not find relevant information in your uploaded documents to answer this question. Please upload a document containing details on this topic or rephrase your query.";
      const finalContent = streamedText.trim() || finalData?.answer || fallbackText;

      const assistantMsg = {
        role: 'assistant',
        content: finalContent,
        citations: finalData?.retrieved_chunks || [],
        faithfulness: finalData?.faithfulness_score,
        relevance: finalData?.relevance_score,
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Request contextual follow-up questions
      fetchFollowups(question, finalContent);
    } catch (err) {
      setError(err.message);
      const errorFallbackMsg = {
        role: 'assistant',
        content: `⚠️ ${err.message}\n\nI could not find relevant information in your uploaded documents to answer this question. Please make sure a relevant document is uploaded or select a valid LLM provider in the sidebar.`,
        citations: [],
      };
      setMessages((prev) => [...prev, errorFallbackMsg]);
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

  const handleClearHistory = () => {
    setMessages([]);
    localStorage.removeItem('rag_chat_messages');
  };

  const handleDeleteAllData = async () => {
    try {
      await fetch(`${API_BASE}/tenant`, {
        method: 'DELETE',
        credentials: 'include',
      });
      setMessages([]);
      localStorage.removeItem('rag_chat_messages');
      setSelectedCitation(null);
      await refreshData();
    } catch (err) {
      setError(`Failed to wipe data: ${err.message}`);
    }
  };

  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [mobileSourceOpen, setMobileSourceOpen] = useState(false);

  const handleSelectCitation = (citation) => {
    setSelectedCitation(citation);
    if (window.innerWidth < 768) {
      setMobileSourceOpen(true);
    }
  };

  return (
    <div className="flex flex-col md:flex-row flex-1 min-h-0 overflow-hidden relative">
      {/* Mobile Top Workspace Bar (Visible on < md) */}
      <div className="md:hidden shrink-0 bg-parchment-200/90 border-b border-warmborder px-3 py-2 flex items-center justify-between text-xs font-sans font-medium text-charcoal-800 select-none shadow-2xs z-20">
        <button
          type="button"
          onClick={() => setMobileSidebarOpen(true)}
          aria-label="Open Controls & Document Uploader"
          className="flex items-center gap-1.5 min-h-[44px] px-3 py-1.5 rounded-xl bg-parchment-50 border border-warmborder text-charcoal-900 hover:border-terracotta-600 shadow-2xs transition-colors cursor-pointer"
        >
          <Sliders className="w-4 h-4 text-terracotta-700 shrink-0" />
          <span>Controls & Upload</span>
          {documents.length > 0 && (
            <span className="px-1.5 py-0.5 rounded-full bg-terracotta-100 text-terracotta-700 font-mono text-[10px] font-bold">
              {documents.length}
            </span>
          )}
        </button>

        <button
          type="button"
          onClick={() => setMobileSourceOpen(true)}
          aria-label="Open Source Inspector Panel"
          className={`flex items-center gap-1.5 min-h-[44px] px-3 py-1.5 rounded-xl border text-xs font-medium transition-colors cursor-pointer ${
            selectedCitation
              ? 'bg-terracotta-100 border-terracotta-600/40 text-terracotta-700 font-semibold'
              : 'bg-parchment-50 border-warmborder text-charcoal-800 hover:text-charcoal-900'
          }`}
        >
          <Bookmark className="w-4 h-4 text-terracotta-700 shrink-0" />
          <span>{selectedCitation ? `Source [${selectedCitation.id || 1}]` : 'Sources'}</span>
        </button>
      </div>

      <Sidebar
        tenantId={tenantId}
        documents={documents}
        stats={stats}
        messages={messages}
        onUpload={handleUpload}
        fileProgress={fileProgress}
        onDeleteDoc={handleDeleteDoc}
        onDeleteAllData={handleDeleteAllData}
        onClearHistory={handleClearHistory}
        onSelectCheckpoint={handleSelectCheckpoint}
        settings={settings}
        onUpdateSettings={setSettings}
      />

      <ReadingPane
        messages={messages}
        onSendMessage={handleSendMessage}
        isStreaming={isStreaming}
        currentStreamText={currentStreamText}
        onSelectCitation={handleSelectCitation}
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

      {/* Mobile Sidebar Sheet / Drawer */}
      {mobileSidebarOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex flex-col justify-end">
          <div
            onClick={() => setMobileSidebarOpen(false)}
            className="fixed inset-0 bg-charcoal-900/50 backdrop-blur-xs transition-opacity animate-fadeIn"
          />
          <div className="fixed inset-x-0 bottom-0 top-12 bg-parchment-200 rounded-t-2xl shadow-2xl z-50 flex flex-col overflow-hidden animate-slideInBottom">
            <Sidebar
              tenantId={tenantId}
              documents={documents}
              stats={stats}
              messages={messages}
              onUpload={handleUpload}
              fileProgress={fileProgress}
              onDeleteDoc={handleDeleteDoc}
              onDeleteAllData={handleDeleteAllData}
              onClearHistory={handleClearHistory}
              onSelectCheckpoint={(idx) => {
                handleSelectCheckpoint(idx);
                setMobileSidebarOpen(false);
              }}
              settings={settings}
              onUpdateSettings={setSettings}
              isMobileDrawer={true}
              onCloseMobile={() => setMobileSidebarOpen(false)}
            />
          </div>
        </div>
      )}

      {/* Mobile Source Inspector Sheet / Drawer */}
      {mobileSourceOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex flex-col justify-end">
          <div
            onClick={() => setMobileSourceOpen(false)}
            className="fixed inset-0 bg-charcoal-900/50 backdrop-blur-xs transition-opacity animate-fadeIn"
          />
          <div className="fixed inset-x-0 bottom-0 top-16 bg-parchment-200 rounded-t-2xl shadow-2xl z-50 flex flex-col overflow-hidden animate-slideInBottom">
            <SourceInspector
              selectedCitation={selectedCitation}
              onClose={() => setMobileSourceOpen(false)}
              debugScores={settings.debugScores}
              isMobileDrawer={true}
            />
          </div>
        </div>
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
