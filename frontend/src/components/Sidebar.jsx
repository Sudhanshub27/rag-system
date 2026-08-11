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
  Server,
} from 'lucide-react';
import ConfirmModal from './ConfirmModal';

function UploadProgress({ fileProgress }) {
  const fileEntries = Object.entries(fileProgress);
  if (fileEntries.length === 0) return null;

  return (
    <div className="space-y-2 mt-2">
      {fileEntries.map(([filename, data]) => {
        let percent = 0;
        let label = 'Processing...';

        if (data.event === 'chunking') {
          percent = 40;
          label = 'Chunking text...';
        } else if (data.event === 'embedding') {
          percent = 70;
          label = 'Generating embeddings...';
        } else if (data.event === 'completed') {
          percent = 100;
          label = 'Indexed!';
        } else if (data.event === 'error') {
          return (
            <div
              key={filename}
              className="text-xs p-2 bg-red-100/60 border border-red-300 text-red-800 rounded font-sans"
            >
              <strong>{filename}:</strong> {data.message || 'Upload failed'}
            </div>
          );
        }

        return (
          <div
            key={filename}
            className="p-2 bg-parchment-50 border border-warmborder rounded text-xs space-y-1 font-sans shadow-2xs"
          >
            <div className="flex justify-between items-center text-charcoal-900 font-medium">
              <span className="truncate max-w-[140px] font-sans">{filename}</span>
              <span className="text-[10px] text-terracotta-600 font-semibold">{percent}%</span>
            </div>
            <div className="w-full bg-parchment-200 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-terracotta-600 h-full transition-all duration-300 rounded-full"
                style={{ width: `${percent}%` }}
              ></div>
            </div>
            <div className="text-[10px] text-charcoal-500 italic">{label}</div>
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

  return (
    <>
      <aside className="w-80 h-full bg-parchment-200 border-r border-warmborder flex flex-col font-sans select-none shrink-0 z-10 shadow-xs">
        {/* Header Branding */}
        <div className="p-4 border-b border-warmborder flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-terracotta-600" />
            <h1 className="font-serif font-bold text-lg text-charcoal-900 tracking-tight">
              Ask My Documents
            </h1>
          </div>
        </div>

        {/* Inline Tenant ID Bar */}
        <div className="px-4 py-2 bg-parchment-50/60 border-b border-warmborder flex items-center justify-between text-xs font-mono text-charcoal-700">
          <div className="flex items-center gap-1.5 overflow-hidden pr-2">
            <Lock className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
            <span className="font-sans font-semibold text-charcoal-500 shrink-0">Tenant:</span>
            <span className="truncate text-charcoal-900 select-all" title={tenantId}>
              {showTenantId ? tenantId : maskedTenantId}
            </span>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={() => setShowTenantId(!showTenantId)}
              className="p-1 hover:bg-parchment-200 text-charcoal-500 hover:text-charcoal-900 rounded transition-colors"
              title={showTenantId ? 'Hide Tenant ID' : 'Show Tenant ID'}
            >
              {showTenantId ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            </button>
            <button
              onClick={handleCopyTenantId}
              className="p-1 hover:bg-parchment-200 text-charcoal-500 hover:text-terracotta-600 rounded transition-colors"
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
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* LLM Engine & Provider Selection */}
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans flex items-center justify-between">
              <span className="flex items-center gap-1">
                <Server className="w-3.5 h-3.5 text-terracotta-600" /> Default LLM Engine
              </span>
              <span className="text-[10px] text-sage-700 font-semibold bg-sage-100/80 px-1.5 py-0.5 rounded">Zero Training</span>
            </div>
            <div className="w-full text-xs bg-parchment-50 border border-warmborder rounded-lg p-2.5 text-charcoal-900 font-sans font-medium shadow-2xs flex items-center justify-between">
              <span>Groq API (Llama 3.3 70B)</span>
              <span className="text-[10px] text-terracotta-600 font-semibold">Active</span>
            </div>
            <div className="text-[10px] text-charcoal-600 bg-parchment-50/80 border border-warmborder p-2 rounded-md font-sans italic leading-tight">
              ⚡ <strong>Groq API Active:</strong> Groq's official API Terms guarantee your document data is <strong>NEVER stored or used for AI model training</strong>.
            </div>
          </div>

          {/* BYOK: Add Your Own API Key */}
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans flex items-center justify-between">
              <span className="flex items-center gap-1">
                <Key className="w-3.5 h-3.5 text-terracotta-600" /> Add Your Own API Key (BYOK)
              </span>
              <span className="text-[10px] text-charcoal-500 font-normal italic">Optional</span>
            </div>
            
            {/* Supported Popular Providers */}
            <select
              value={settings.provider || 'groq'}
              onChange={handleProviderChange}
              className="w-full text-xs bg-parchment-50 border border-warmborder rounded-lg p-2 text-charcoal-900 font-sans focus:outline-none focus:border-terracotta-600 shadow-2xs"
            >
              <option value="groq">Groq API (Free & Paid Keys)</option>
              <option value="openai">OpenAI API (All GPT Models)</option>
              <option value="anthropic">Anthropic Claude API (All Claude Models)</option>
              <option value="deepseek">DeepSeek API (All DeepSeek Models)</option>
              <option value="gemini">Google Gemini API (All Gemini Models)</option>
              <option value="openrouter">OpenRouter API (All OpenRouter Models)</option>
            </select>

            <div className="relative flex items-center">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={settings.apiKey || ''}
                onChange={handleApiKeyChange}
                placeholder={`Paste ${settings.provider ? settings.provider.toUpperCase() : 'GROQ'} API key...`}
                className="w-full text-xs bg-parchment-50 border border-warmborder rounded-lg p-2 pr-7 text-charcoal-900 font-mono placeholder:text-charcoal-500/60 focus:outline-none focus:border-terracotta-600 shadow-2xs"
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-2 text-charcoal-500 hover:text-charcoal-900"
              >
                {showApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          {/* Pipeline & Privacy Controls */}
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-terracotta-600" /> Pipeline & Privacy Options
            </div>
            
            {/* Permanent PII Privacy Notice */}
            <div className="p-2.5 bg-sage-50/80 border border-sage-200 rounded-lg text-xs text-charcoal-900 font-sans space-y-1 shadow-2xs">
              <div className="font-semibold flex items-center gap-1 text-sage-900">
                <ShieldCheck className="w-4 h-4 text-sage-700 shrink-0" />
                Local PII Privacy Active
              </div>
              <p className="text-[10px] text-charcoal-600 italic leading-tight">
                All names, emails, phone numbers, and IP addresses are automatically parsed and redacted locally before any data leaves your browser.
              </p>
            </div>

            <div className="space-y-1.5 text-xs text-charcoal-900 bg-parchment-50/70 border border-warmborder p-3 rounded-lg shadow-2xs">
              <label className="flex items-center gap-2 cursor-pointer hover:text-terracotta-600 transition-colors">
                <input
                  type="checkbox"
                  checked={settings.splitView}
                  onChange={() => handleCheckboxChange('splitView')}
                  className="w-3.5 h-3.5 accent-terracotta-600 rounded cursor-pointer"
                />
                <span>Split-Screen Source Inspector</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer hover:text-terracotta-600 transition-colors">
                <input
                  type="checkbox"
                  checked={settings.debugScores}
                  onChange={() => handleCheckboxChange('debugScores')}
                  className="w-3.5 h-3.5 accent-terracotta-600 rounded cursor-pointer"
                />
                <span>Debug Retrieval Scores</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer hover:text-terracotta-600 transition-colors">
                <input
                  type="checkbox"
                  checked={settings.useHyde}
                  onChange={() => handleCheckboxChange('useHyde')}
                  className="w-3.5 h-3.5 accent-terracotta-600 rounded cursor-pointer"
                />
                <span>HyDE Retrieval</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer hover:text-terracotta-600 transition-colors">
                <input
                  type="checkbox"
                  checked={settings.useMultiQuery}
                  onChange={() => handleCheckboxChange('useMultiQuery')}
                  className="w-3.5 h-3.5 accent-terracotta-600 rounded cursor-pointer"
                />
                <span>Multi-Query Expansion</span>
              </label>
            </div>
          </div>

          {/* Drag & Drop File Upload */}
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans">
              Upload Documents
            </div>
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`border border-dashed rounded-lg p-4 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-terracotta-600 bg-parchment-50'
                  : 'border-warmborder hover:border-terracotta-600/60 bg-parchment-50/70'
              }`}
            >
              <input
                type="file"
                multiple
                accept=".pdf,.txt,.md"
                onChange={handleFileInput}
                className="hidden"
                id="sidebar-file-input"
              />
              <label htmlFor="sidebar-file-input" className="cursor-pointer flex flex-col items-center gap-1.5">
                <Upload className="w-5 h-5 text-terracotta-600" />
                <div className="text-xs text-charcoal-700">
                  <span className="font-semibold text-terracotta-600">Click to upload</span> or drag PDFs/TXT
                </div>
                <div className="text-[10px] text-charcoal-500">PDF, TXT, Markdown supported</div>
              </label>
            </div>
            <UploadProgress fileProgress={fileProgress} />
          </div>

          {/* Recent Questions Session History */}
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans flex items-center justify-between">
              <span className="flex items-center gap-1">
                <History className="w-3.5 h-3.5 text-terracotta-600" /> Recent Questions
              </span>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-charcoal-500 font-mono">({recentQuestions.length})</span>
                {recentQuestions.length > 0 && onClearHistory && (
                  <button
                    onClick={onClearHistory}
                    title="Clear Question History"
                    className="p-0.5 text-charcoal-400 hover:text-rust-600 transition-colors"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                )}
              </div>
            </div>

            {recentQuestions.length === 0 ? (
              <div className="text-xs text-charcoal-500 italic p-3 bg-parchment-50/50 rounded-lg border border-warmborder/60">
                No questions asked yet in this session.
              </div>
            ) : (
              <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
                {recentQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => onSelectCheckpoint(q.originalIndex)}
                    className="w-full text-left p-2 rounded-md bg-parchment-50 hover:bg-parchment-300/40 border border-warmborder text-xs text-charcoal-900 truncate font-serif italic flex items-center gap-2 transition-colors shadow-2xs"
                  >
                    <MessageSquare className="w-3 h-3 text-terracotta-600 shrink-0 not-italic" />
                    <span className="truncate">{q.content}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Knowledge Stats */}
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans">
              Corpus Stats
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-parchment-50 border border-warmborder p-2.5 rounded-lg shadow-2xs">
                <div className="text-xl font-serif font-bold text-charcoal-900">{stats.total_chunks || 0}</div>
                <div className="text-[11px] text-charcoal-500 flex items-center gap-1 font-sans">
                  <BookOpen className="w-3 h-3 text-terracotta-600" /> Total Chunks
                </div>
              </div>
              <div className="bg-parchment-50 border border-warmborder p-2.5 rounded-lg shadow-2xs">
                <div className="text-xl font-serif font-bold text-charcoal-900">{stats.total_documents || 0}</div>
                <div className="text-[11px] text-charcoal-500 flex items-center gap-1 font-sans">
                  <BookOpen className="w-3 h-3 text-terracotta-600" /> Documents
                </div>
              </div>
            </div>
          </div>

          {/* Indexed Documents List */}
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans flex justify-between items-center">
              <span>Indexed Documents</span>
              <span className="text-[10px] text-charcoal-500 font-mono">({documents.length})</span>
            </div>
            {documents.length === 0 ? (
              <div className="text-xs text-charcoal-500 italic p-3 bg-parchment-50/50 rounded-lg border border-warmborder/60">
                No documents indexed yet. Upload a file above.
              </div>
            ) : (
              <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
                {documents.map((doc, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-2 rounded-md bg-parchment-50 border border-warmborder text-xs text-charcoal-900 font-sans shadow-2xs"
                  >
                    <span className="truncate max-w-[170px]" title={doc.filename || doc}>
                      {doc.filename || doc}
                    </span>
                    <button
                      onClick={() => promptDeleteDocument(doc.filename || doc)}
                      className="p-1 text-charcoal-500 hover:text-rust-600 transition-colors"
                      title="Delete document"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer Data Control */}
        <div className="p-4 border-t border-warmborder bg-parchment-200">
          <button
            onClick={promptDeleteAllData}
            className="w-full py-2 px-3 text-xs font-semibold text-rust-600 hover:text-white bg-parchment-50 hover:bg-rust-600 border border-rust-600/30 rounded-lg transition-all shadow-2xs flex items-center justify-center gap-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" /> Delete all my data
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
