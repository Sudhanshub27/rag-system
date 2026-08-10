import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import SourceInspector from './components/SourceInspector';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
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

  // SSE Answer Streaming Query
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
        body: JSON.stringify({ question }),
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
    if (!window.confirm('Are you sure you want to permanently delete all your uploaded documents?')) return;
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
    <div className="flex h-screen w-screen overflow-hidden bg-zinc-950 font-sans antialiased">
      <Sidebar
        tenantId={tenantId}
        documents={documents}
        stats={stats}
        onUpload={handleUpload}
        fileProgress={fileProgress}
        onDeleteDoc={handleDeleteDoc}
        onDeleteAllData={handleDeleteAllData}
      />
      <ChatPanel
        messages={messages}
        onSendMessage={handleSendMessage}
        isStreaming={isStreaming}
        currentStreamText={currentStreamText}
        onSelectCitation={setSelectedCitation}
        error={error}
        followups={followups}
        onSelectFollowup={handleSendMessage}
      />
      <SourceInspector
        selectedCitation={selectedCitation}
        onClose={() => setSelectedCitation(null)}
      />
    </div>
  );
}
