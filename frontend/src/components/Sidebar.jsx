import React, { useState } from 'react';
import {
  Upload,
  BookOpen,
  Trash2,
  Lock,
  MessageSquare,
  History,
  Eye,
  EyeOff,
  Copy,
  Check,
  Cpu,
  Key,
  ShieldCheck,
  Sliders,
  FileText,
  Sparkles,
  Zap,
  CheckCircle2,
  ChevronRight,
  Info,
  X,
} from 'lucide-react';
import ConfirmModal from './ConfirmModal';

function UploadProgress({ fileProgress }) {
  const fileEntries = Object.entries(fileProgress);
  if (fileEntries.length === 0) return null;

  return (
    <div className="space-y-2 mt-2">
      {fileEntries.map(([filename, data]) => {
        let percent = data.progress || 0;
        let label = data.message || 'Processing...';

        const isComplete =
          data.event === 'complete' ||
          data.event === 'completed' ||
          data.stage === 'complete' ||
          percent === 100;
        const isError = data.event === 'error' || data.stage === 'error';
        const isIndexing = data.event === 'indexing' || data.stage === 'indexing';
        const isEmbedding = data.event === 'embedding' || data.stage === 'embedding';
        const isChunking = data.event === 'chunking' || data.stage === 'chunking';

        if (isError) {
          return (
            <div
              key={filename}
              className="text-xs p-2.5 bg-red-500/10 border border-red-500/30 text-red-800 rounded-xl font-sans flex items-center gap-2"
            >
              <div className="w-2 h-2 rounded-full bg-red-600 shrink-0 animate-ping" />
              <div className="truncate">
                <strong className="font-semibold">{filename}:</strong> {data.error || data.message || 'Upload failed'}
              </div>
            </div>
          );
        }

        if (isComplete) {
          percent = 100;
          label = data.chunks_added
            ? `Indexed! (${data.chunks_added} chunks)`
            : 'Indexed & Ready!';
        } else if (isIndexing) {
          percent = 85;
          label = 'Storing in ChromaDB...';
        } else if (isEmbedding) {
          percent = 60;
          label = 'Generating embeddings...';
        } else if (isChunking) {
          percent = 25;
          label = 'Chunking text...';
        }

        return (
          <div
            key={filename}
            className="p-2.5 bg-parchment-50/90 border border-warmborder/80 rounded-xl text-xs space-y-1.5 font-sans shadow-2xs transition-all"
          >
            <div className="flex justify-between items-center text-charcoal-900 font-medium">
              <span className="truncate max-w-[150px] font-sans flex items-center gap-1.5" title={filename}>
                <FileText className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
                <span className="truncate font-semibold">{filename}</span>
              </span>
              <span
                className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                  isComplete ? 'bg-sage-600/15 text-sage-700' : 'bg-terracotta-600/15 text-terracotta-600 animate-pulse'
                }`}
              >
                {isComplete ? '100% ✓' : `${percent}%`}
              </span>
            </div>
            <div className="w-full bg-parchment-300/60 h-1.5 rounded-full overflow-hidden p-0.5">
              <div
                className={`h-full transition-all duration-300 rounded-full ${
                  isComplete ? 'bg-sage-600' : 'bg-gradient-to-r from-terracotta-500 to-amber-500'
                }`}
                style={{ width: `${percent}%` }}
              ></div>
            </div>
            <div
              className={`text-[10px] flex items-center justify-between ${
                isComplete ? 'text-sage-700 font-semibold' : 'text-charcoal-500 italic'
              }`}
            >
              <span>{label}</span>
              {isComplete && <CheckCircle2 className="w-3 h-3 text-sage-600 shrink-0" />}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Sidebar({
  tenantId,
  documents = [],
  stats = {},
  messages = [],
  onUpload,
  fileProgress = {},
  onDeleteDoc,
  onDeleteAllData,
  onClearHistory,
  onSelectCheckpoint,
  settings = {},
  onUpdateSettings,
  isMobileDrawer = false,
  onCloseMobile = null,
}) {
  const [dragActive, setDragActive] = useState(false);
  const [showTenantId, setShowTenantId] = useState(false);
  const [copiedTenantId, setCopiedTenantId] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  // Modal State for Confirmations
  const [confirmModal, setConfirmModal] = useState({
    isOpen: false,
    title: '',
    message: '',
    confirmText: 'Delete',
    onConfirm: null,
  });

  const handleCheckboxChange = (key) => {
    onUpdateSettings((prev) => {
      const updated = { ...prev, [key]: !prev[key] };
      if (key === 'anonymizePii') {
        localStorage.setItem('rag_anonymize_pii', updated[key]);
      }
      return updated;
    });
  };

  const handleProviderChange = (e) => {
    const provider = e.target.value;
    onUpdateSettings((prev) => {
      const updated = { ...prev, provider };
      localStorage.setItem('rag_provider', provider);
      return updated;
    });
  };

  const handleApiKeyChange = (e) => {
    const apiKey = e.target.value;
    onUpdateSettings((prev) => {
      const updated = { ...prev, apiKey };
      localStorage.setItem('rag_api_key', apiKey);
      return updated;
    });
  };

  const handleModelChange = (model) => {
    onUpdateSettings((prev) => {
      const updated = { ...prev, model };
      localStorage.setItem('rag_model', model);
      return updated;
    });
  };

  const handleCopyTenantId = () => {
    if (!tenantId) return;
    navigator.clipboard.writeText(tenantId);
    setCopiedTenantId(true);
    setTimeout(() => setCopiedTenantId(false), 2000);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(Array.from(e.target.files));
    }
  };

  const promptDeleteDocument = (docName) => {
    setConfirmModal({
      isOpen: true,
      title: 'Delete Document',
      message: `Are you sure you want to delete "${docName}"? It will be permanently removed from your knowledge index.`,
      confirmText: 'Delete Document',
      onConfirm: () => {
        onDeleteDoc(docName);
        setConfirmModal((prev) => ({ ...prev, isOpen: false }));
      },
    });
  };

  const promptDeleteAllData = () => {
    setConfirmModal({
      isOpen: true,
      title: 'Wipe All Data',
      message:
        'Are you sure you want to permanently delete all uploaded documents and reset your vector database? This action cannot be undone.',
      confirmText: 'Delete All Data',
      onConfirm: () => {
        onDeleteAllData();
        setConfirmModal((prev) => ({ ...prev, isOpen: false }));
      },
    });
  };

  const maskedTenantId = tenantId
    ? `${tenantId.slice(0, 8)}-••••-••••-••••-••••••••••••`
    : 'Generating...';

  // Extract recent user questions for checkpoint navigation (newest first)
  const recentQuestions = messages
    .map((msg, index) => ({ ...msg, originalIndex: index }))
    .filter((msg) => msg.role === 'user')
    .slice()
    .reverse();

  // Custom Toggle Switch Component
  const ToggleSwitch = ({ checked, onChange, label, description }) => (
    <div
      onClick={onChange}
      className="group flex items-center justify-between p-2 rounded-xl hover:bg-parchment-200/50 transition-all cursor-pointer select-none"
    >
      <div className="flex flex-col pr-2">
        <span className="text-xs font-medium text-charcoal-900 group-hover:text-terracotta-700 transition-colors">
          {label}
        </span>
        {description && (
          <span className="text-[10px] text-charcoal-500/80 leading-tight">
            {description}
          </span>
        )}
      </div>
      <div
        className={`w-8 h-4.5 flex items-center rounded-full p-0.5 transition-colors duration-200 shrink-0 ${
          checked ? 'bg-terracotta-600 shadow-xs' : 'bg-charcoal-300/60'
        }`}
      >
        <div
          className={`bg-white w-3.5 h-3.5 rounded-full shadow-md transform transition-transform duration-200 ${
            checked ? 'translate-x-3.5' : 'translate-x-0'
          }`}
        />
      </div>
    </div>
  );

  return (
    <>
      <aside className={`${isMobileDrawer ? 'w-full h-full' : 'hidden md:flex w-80 h-full border-r'} bg-parchment-200/90 border-warmborder flex flex-col font-sans select-none shrink-0 z-10 shadow-sm backdrop-blur-xs`}>
        {isMobileDrawer && (
          <div className="px-4 py-3 bg-parchment-100 border-b border-warmborder flex items-center justify-between font-sans shrink-0">
            <div className="flex items-center gap-2 font-bold text-sm text-charcoal-900">
              <Sliders className="w-4 h-4 text-terracotta-600" />
              <span>RAG Controls & Upload</span>
            </div>
            <button
              type="button"
              onClick={onCloseMobile}
              className="flex items-center justify-center min-w-[44px] min-h-[44px] p-2 text-charcoal-500 hover:text-charcoal-900 rounded-xl hover:bg-parchment-200 transition-colors cursor-pointer"
              aria-label="Close Controls"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        )}
        {/* Inline Tenant Session Bar at Top */}
        <div className="px-3.5 py-2.5 bg-parchment-50/90 border-b border-warmborder/80 flex items-center justify-between text-xs font-mono text-charcoal-700 shadow-2xs">
          <div className="flex items-center gap-1.5 overflow-hidden pr-2">
            <div className="p-1 rounded-md bg-terracotta-600/10 border border-terracotta-600/20 text-terracotta-600">
              <Lock className="w-3.5 h-3.5 shrink-0" />
            </div>
            <span className="font-sans font-semibold text-charcoal-500 shrink-0">Tenant:</span>
            <span className="truncate text-charcoal-900 font-medium select-all" title={tenantId}>
              {showTenantId ? tenantId : maskedTenantId}
            </span>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={() => setShowTenantId(!showTenantId)}
              className="p-1.5 hover:bg-parchment-200 text-charcoal-500 hover:text-charcoal-900 rounded-lg transition-colors cursor-pointer"
              title={showTenantId ? 'Hide Tenant ID' : 'Show Tenant ID'}
            >
              {showTenantId ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            </button>
            <button
              onClick={handleCopyTenantId}
              className="p-1.5 hover:bg-parchment-200 text-charcoal-500 hover:text-terracotta-600 rounded-lg transition-colors cursor-pointer"
              title="Copy Tenant ID"
            >
              {copiedTenantId ? (
                <Check className="w-3.5 h-3.5 text-sage-600" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
        </div>

        {/* Scrollable Main Options Area */}
        <div className="flex-1 overflow-y-auto p-3.5 space-y-4">

          {/* LLM Engine Selection */}
          <div className="space-y-2">
            <div className="text-[11px] font-bold text-charcoal-600 uppercase tracking-wider font-sans flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <Key className="w-3.5 h-3.5 text-terracotta-600" /> LLM Provider & Key
              </span>
            </div>
            
            <div className="space-y-2 bg-parchment-50/90 border border-warmborder p-3 rounded-xl shadow-2xs">
              <div className="space-y-1">
                <label className="text-[10px] font-semibold uppercase text-charcoal-500 tracking-wider">Provider</label>
                <select
                  value={settings.provider || 'groq'}
                  onChange={handleProviderChange}
                  className="w-full text-xs bg-white border border-warmborder/90 rounded-lg p-2 text-charcoal-900 font-sans focus:outline-none focus:border-terracotta-600 focus:ring-2 focus:ring-terracotta-600/15 shadow-2xs cursor-pointer font-medium"
                >
                  <option value="groq">Groq API (Default - Free 70B & Zero-Training)</option>
                  <option value="openai">OpenAI API</option>
                  <option value="anthropic">Anthropic Claude API</option>
                  <option value="deepseek">DeepSeek API</option>
                  <option value="gemini">Google Gemini API</option>
                  <option value="openrouter">OpenRouter API</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[10px] font-semibold uppercase text-charcoal-500 tracking-wider">API Key</label>
                <div className="relative flex items-center">
                  <input
                    type={showApiKey ? 'text' : 'password'}
                    value={settings.apiKey || ''}
                    onChange={handleApiKeyChange}
                    placeholder={settings.provider === 'groq' || !settings.provider ? "Groq active (Optional custom key)..." : `Paste ${settings.provider ? settings.provider.toUpperCase() : 'API'} key...`}
                    className="w-full text-xs bg-white border border-warmborder/90 rounded-lg p-2 pr-8 text-charcoal-900 font-mono placeholder:text-charcoal-400 focus:outline-none focus:border-terracotta-600 focus:ring-2 focus:ring-terracotta-600/15 shadow-2xs"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2 text-charcoal-400 hover:text-charcoal-900 p-1 rounded transition-colors cursor-pointer"
                  >
                    {showApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              {/* Optional Model Override */}
              <div className="space-y-1">
                <label className="text-[10px] font-semibold uppercase text-charcoal-500 tracking-wider">Optional Model ID Override</label>
                <input
                  type="text"
                  value={settings.model || ''}
                  onChange={(e) => handleModelChange(e.target.value)}
                  placeholder="e.g. claude-3-5-sonnet, gpt-4o, o3-mini..."
                  className="w-full text-[11px] bg-white border border-warmborder/90 rounded-lg p-1.5 text-charcoal-900 font-mono placeholder:text-charcoal-400 focus:outline-none focus:border-terracotta-600 focus:ring-2 focus:ring-terracotta-600/15 shadow-2xs"
                />
              </div>

              {/* Live API Rate Limit & Protection Pill */}
              <div className="pt-1 flex items-center justify-between text-[10px] bg-terracotta-500/5 border border-terracotta-600/15 rounded-lg px-2.5 py-1.5">
                <span className="flex items-center gap-1.5 font-mono text-terracotta-700 font-semibold">
                  <ShieldCheck className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
                  <span>Quota Shield: 6 req/m • 50 req/h</span>
                </span>
                <span className="flex items-center gap-1 text-sage-700 font-bold">
                  <span className="w-1.5 h-1.5 rounded-full bg-sage-600 animate-pulse" />
                  Active
                </span>
              </div>
            </div>
          </div>

          {/* Pipeline Options Controls */}
          <div className="space-y-2">
            <div className="text-[11px] font-bold text-charcoal-600 uppercase tracking-wider font-sans flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-terracotta-600" /> RAG Pipeline Controls
            </div>

            <div className="bg-parchment-50/90 border border-warmborder p-2 rounded-xl shadow-2xs divide-y divide-warmborder/40">
              <ToggleSwitch
                checked={settings.splitView}
                onChange={() => handleCheckboxChange('splitView')}
                label="Split-Screen Inspector"
                description="Side-by-side context document viewer"
              />
              <ToggleSwitch
                checked={settings.debugScores}
                onChange={() => handleCheckboxChange('debugScores')}
                label="Debug Retrieval Scores"
                description="Display BM25 & vector similarity metrics"
              />
              <ToggleSwitch
                checked={settings.useHyde}
                onChange={() => handleCheckboxChange('useHyde')}
                label="HyDE Retrieval"
                description="Hypothetical document embeddings"
              />
              <ToggleSwitch
                checked={settings.useMultiQuery}
                onChange={() => handleCheckboxChange('useMultiQuery')}
                label="Multi-Query Expansion"
                description="Generate 3 query variations automatically"
              />
            </div>

            {/* Smart Large Document Recommendation Banner (Parchment & Terracotta Theme) */}
            {stats && stats.total_chunks > 20 && (!settings.useHyde || !settings.useMultiQuery) && (
              <div className="p-3 bg-parchment-50 border border-warmborder rounded-xl space-y-2 font-sans shadow-2xs">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-charcoal-900">
                    <Sparkles className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
                    <span>Large Document Detected</span>
                  </div>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-terracotta-100 text-terracotta-700 border border-terracotta-600/30">
                    {stats.total_chunks} Chunks
                  </span>
                </div>

                <p className="text-[11px] text-charcoal-700 leading-tight">
                  For maximum search accuracy across multi-page documents, enabling <strong>HyDE</strong> & <strong>Multi-Query</strong> mode is strongly recommended.
                </p>

                <div className="flex items-center gap-1.5 pt-0.5">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-parchment-200 text-charcoal-700 border border-warmborder flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-sage-600" /> HyDE Search
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-parchment-200 text-charcoal-700 border border-warmborder flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-sage-600" /> Multi-Query 3x
                  </span>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    onUpdateSettings((prev) => ({ ...prev, useHyde: true, useMultiQuery: true }));
                  }}
                  className="w-full text-xs font-semibold bg-terracotta-600 hover:bg-terracotta-700 text-white p-2 rounded-lg transition-colors shadow-2xs flex items-center justify-center gap-1.5 cursor-pointer mt-1"
                >
                  <Zap className="w-3.5 h-3.5 text-white shrink-0" />
                  Enable HyDE & Multi-Query Mode
                </button>
              </div>
            )}
          </div>

          {/* Drag & Drop File Upload */}
          <div className="space-y-2">
            <div className="text-[11px] font-bold text-charcoal-600 uppercase tracking-wider font-sans flex justify-between items-center">
              <span>Upload Documents</span>
            </div>
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-3.5 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-terracotta-600 bg-parchment-50 shadow-inner'
                  : 'border-warmborder hover:border-terracotta-600/60 bg-parchment-50/80 hover:bg-parchment-50 shadow-2xs'
              }`}
            >
              <input
                type="file"
                multiple
                accept=".pdf,.txt,.doc,.docx,.csv,.tsv,.xls,.xlsx,.md,.markdown,.json,.jsonl,.html,.htm,.rst,.xml,.yaml,.yml,.log"
                onChange={handleFileInput}
                className="hidden"
                id="sidebar-file-input"
              />
              <label htmlFor="sidebar-file-input" className="cursor-pointer flex flex-col items-center gap-1.5">
                <div className="p-2 rounded-full bg-terracotta-600/10 text-terracotta-600">
                  <Upload className="w-4 h-4" />
                </div>
                <div className="text-xs text-charcoal-700">
                  <span className="font-bold text-terracotta-700 hover:underline">Click to upload</span> or drag files
                </div>
                <div className="text-[10px] text-charcoal-500 font-mono">PDF, Word, Excel, CSV, JSON, MD, TXT</div>
              </label>
            </div>
            <UploadProgress fileProgress={fileProgress} />
          </div>

          {/* Knowledge Index Statistics Dashboard Banner */}
          <div className="p-3 bg-parchment-50/90 border border-warmborder rounded-xl shadow-2xs flex items-center justify-between font-sans">
            <div className="flex items-center gap-2 text-xs font-bold text-charcoal-800">
              <div className="p-1 rounded-md bg-terracotta-600/10 text-terracotta-600">
                <Cpu className="w-3.5 h-3.5" />
              </div>
              <span>Knowledge Index</span>
            </div>
            <div className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="px-2 py-0.5 rounded-md bg-parchment-200 text-charcoal-900 font-semibold border border-warmborder/80">
                {stats.total_documents ?? documents.length ?? 0} Docs
              </span>
              <span className="px-2 py-0.5 rounded-md bg-terracotta-600/15 text-terracotta-700 font-bold border border-terracotta-600/30">
                {stats.total_chunks ?? 0} Chunks
              </span>
            </div>
          </div>

          {/* Indexed Documents List */}
          <div className="space-y-2">
            <div className="text-[11px] font-bold text-charcoal-600 uppercase tracking-wider font-sans flex justify-between items-center">
              <span>Indexed Documents</span>
              <span className="text-[10px] text-charcoal-500 font-mono font-semibold">({documents.length})</span>
            </div>
            {documents.length === 0 ? (
              <div className="text-xs text-charcoal-500 italic p-3 bg-parchment-50/60 rounded-xl border border-warmborder/60 text-center">
                No documents indexed yet. Upload a file above.
              </div>
            ) : (
              <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                {documents.map((doc, idx) => (
                  <div
                    key={idx}
                    className="group flex items-center justify-between p-2 px-2.5 rounded-lg bg-parchment-50 border border-warmborder text-xs text-charcoal-900 font-sans shadow-2xs hover:border-terracotta-600/40 transition-colors"
                  >
                    <span className="truncate max-w-[170px] flex items-center gap-2" title={doc.filename || doc}>
                      <FileText className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
                      <span className="truncate font-medium">{doc.filename || doc}</span>
                    </span>
                    <button
                      onClick={() => promptDeleteDocument(doc.filename || doc)}
                      className="p-1 text-charcoal-400 hover:text-rust-600 rounded transition-colors opacity-80 group-hover:opacity-100 cursor-pointer"
                      title="Delete document"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Recent Questions Session History */}
          <div className="space-y-2">
            <div className="text-[11px] font-bold text-charcoal-600 uppercase tracking-wider font-sans flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <History className="w-3.5 h-3.5 text-terracotta-600" /> Recent Questions
              </span>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-charcoal-500 font-mono font-semibold">({recentQuestions.length})</span>
                {recentQuestions.length > 0 && onClearHistory && (
                  <button
                    onClick={onClearHistory}
                    title="Clear Question History"
                    className="p-1 text-charcoal-400 hover:text-rust-600 rounded transition-colors cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            {recentQuestions.length === 0 ? (
              <div className="text-xs text-charcoal-500 italic p-3 bg-parchment-50/60 rounded-xl border border-warmborder/60 text-center">
                No questions asked yet.
              </div>
            ) : (
              <div className="space-y-1.5 max-h-32 overflow-y-auto pr-1">
                {recentQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => onSelectCheckpoint(q.originalIndex)}
                    className="w-full text-left p-2 px-2.5 rounded-lg bg-parchment-50 hover:bg-parchment-300/50 border border-warmborder text-xs text-charcoal-900 truncate font-serif italic flex items-center justify-between gap-2 transition-all shadow-2xs group cursor-pointer"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <MessageSquare className="w-3.5 h-3.5 text-terracotta-600 shrink-0 not-italic" />
                      <span className="truncate">{q.content}</span>
                    </div>
                    <ChevronRight className="w-3 h-3 text-charcoal-400 group-hover:text-terracotta-600 transition-colors shrink-0 not-italic" />
                  </button>
                ))}
              </div>
            )}
          </div>

        </div>

        {/* Footer Data Control */}
        <div className="p-3 border-t border-warmborder bg-parchment-200/90">
          <button
            onClick={promptDeleteAllData}
            className="w-full py-2 px-3 text-xs font-bold text-rust-600 hover:text-white bg-parchment-50 hover:bg-rust-600 border border-rust-600/30 hover:border-rust-600 rounded-xl transition-all shadow-2xs flex items-center justify-center gap-2 cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" /> Wipe All My Data
          </button>
        </div>
      </aside>

      {/* Parchment Styled Confirmation Modal */}
      <ConfirmModal
        isOpen={confirmModal.isOpen}
        title={confirmModal.title}
        message={confirmModal.message}
        confirmText={confirmModal.confirmText}
        onConfirm={confirmModal.onConfirm}
        onCancel={() => setConfirmModal((prev) => ({ ...prev, isOpen: false }))}
      />
    </>
  );
}
