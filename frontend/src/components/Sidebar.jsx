import React, { useState } from 'react';
import { Upload, Trash2, FileText, BookOpen, AlertTriangle, History, MessageSquare, Sliders, Database, Eye, EyeOff, Copy, Check } from 'lucide-react';
import UploadProgress from './UploadProgress';
import ConfirmModal from './ConfirmModal';

export default function Sidebar({
  tenantId,
  documents,
  stats,
  messages,
  onUpload,
  fileProgress,
  onDeleteDoc,
  onDeleteAllData,
  onSelectCheckpoint,
  settings,
  onUpdateSettings,
}) {
  const [dragActive, setDragActive] = useState(false);
  const [showFullTenantId, setShowFullTenantId] = useState(false);
  const [copiedTenantId, setCopiedTenantId] = useState(false);

  // State for Custom Parchment Confirmation Modal
  const [confirmModal, setConfirmModal] = useState({
    isOpen: false,
    title: '',
    message: '',
    confirmText: 'Delete',
    onConfirm: () => {},
  });

  const handleCopyTenantId = () => {
    if (!tenantId) return;
    navigator.clipboard.writeText(tenantId);
    setCopiedTenantId(true);
    setTimeout(() => setCopiedTenantId(false), 2000);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
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

  const handleCheckboxChange = (key) => {
    onUpdateSettings({
      ...settings,
      [key]: !settings[key],
    });
  };

  const promptDeleteDocument = (filename) => {
    setConfirmModal({
      isOpen: true,
      title: 'Delete Document',
      message: `Are you sure you want to delete "${filename}"? It will be removed from your vector search index immediately.`,
      confirmText: 'Delete Document',
      onConfirm: () => onDeleteDoc(filename),
    });
  };

  const promptDeleteAllData = () => {
    setConfirmModal({
      isOpen: true,
      title: 'Delete All My Data',
      message: 'Are you sure you want to permanently delete all uploaded documents and session history? This action cannot be undone.',
      confirmText: 'Delete All Data',
      onConfirm: () => onDeleteAllData(),
    });
  };

  // Filter user questions for Recent Questions list
  const recentQuestions = messages
    .map((m, idx) => ({ ...m, originalIndex: idx }))
    .filter((m) => m.role === 'user');

  const maskedTenantId = tenantId ? `${tenantId.slice(0, 8)}-••••-••••-••••-••••••••••••` : 'Connecting...';

  return (
    <aside className="w-80 shrink-0 bg-parchment-200 border-r border-warmborder flex flex-col h-full text-charcoal-900 select-none font-sans overflow-hidden">
      {/* Full Tenant ID Section with Eye Toggle */}
      <div className="p-3.5 border-b border-warmborder bg-parchment-200/90 shrink-0 space-y-1.5">
        <div className="flex items-center justify-between text-[11px] font-sans font-semibold text-charcoal-500 uppercase tracking-wider">
          <span className="flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-terracotta-600" /> Tenant ID
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setShowFullTenantId(!showFullTenantId)}
              className="p-1 hover:bg-parchment-50 rounded text-charcoal-500 hover:text-terracotta-600 transition-colors"
              title={showFullTenantId ? 'Hide Tenant ID' : 'Show Full Tenant ID'}
            >
              {showFullTenantId ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            </button>
            <button
              onClick={handleCopyTenantId}
              className="p-1 hover:bg-parchment-50 rounded text-charcoal-500 hover:text-terracotta-600 transition-colors"
              title="Copy Tenant ID"
            >
              {copiedTenantId ? <Check className="w-3.5 h-3.5 text-sage-600" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
        <div className="bg-parchment-50 border border-warmborder rounded-lg p-2.5 font-mono text-xs text-charcoal-900 font-semibold break-all select-text shadow-2xs">
          {showFullTenantId ? (tenantId || 'Connecting...') : maskedTenantId}
        </div>
      </div>

      {/* Scrollable Sidebar Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs">
        {/* Settings & Features */}
        <div className="space-y-2.5 p-3 bg-parchment-50 border border-warmborder rounded-lg shadow-2xs">
          <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-terracotta-600" /> Settings & Features
          </div>
          <div className="space-y-2 text-xs text-charcoal-900 font-sans">
            <label className="flex items-center gap-2 cursor-pointer hover:text-terracotta-600 transition-colors">
              <input
                type="checkbox"
                checked={settings.splitView}
                onChange={() => handleCheckboxChange('splitView')}
                className="w-3.5 h-3.5 accent-terracotta-600 rounded cursor-pointer"
              />
              <span>Split View Inspector</span>
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
            <span className="text-[10px] text-charcoal-500 font-mono">({recentQuestions.length})</span>
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
              <div className="text-xl font-serif font-bold text-charcoal-900">{documents.length || 0}</div>
              <div className="text-[11px] text-charcoal-500 flex items-center gap-1 font-sans">
                <FileText className="w-3 h-3 text-terracotta-600" /> Documents
              </div>
            </div>
          </div>
        </div>

        {/* Index List */}
        <div className="space-y-2">
          <div className="text-[11px] font-semibold text-charcoal-500 uppercase tracking-wider font-sans">
            Your Documents
          </div>
          {documents.length === 0 ? (
            <div className="text-xs text-charcoal-500 italic p-3 bg-parchment-50/50 rounded-lg border border-warmborder/60">
              No files uploaded yet.
            </div>
          ) : (
            <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
              {documents.map((doc) => (
                <div
                  key={doc.filename}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-parchment-50 border border-warmborder text-xs shadow-2xs"
                >
                  <div className="truncate max-w-[160px]">
                    <div className="font-serif font-medium text-charcoal-900 truncate">{doc.filename}</div>
                    <div className="text-[10px] text-charcoal-500">{doc.chunk_count} chunks</div>
                  </div>
                  <button
                    onClick={() => promptDeleteDocument(doc.filename)}
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

      {/* Footer Purge Button */}
      <div className="p-4 border-t border-warmborder bg-parchment-200 shrink-0">
        <button
          onClick={promptDeleteAllData}
          className="w-full py-2 px-3 rounded-lg bg-rust-100/80 hover:bg-rust-100 border border-rust-600/30 text-rust-600 text-xs font-semibold flex items-center justify-center gap-2 transition-colors"
        >
          <AlertTriangle className="w-3.5 h-3.5" />
          Delete all my data
        </button>
      </div>

      {/* Reusable Parchment Styled Confirmation Modal */}
      <ConfirmModal
        isOpen={confirmModal.isOpen}
        title={confirmModal.title}
        message={confirmModal.message}
        confirmText={confirmModal.confirmText}
        onConfirm={confirmModal.onConfirm}
        onClose={() => setConfirmModal((prev) => ({ ...prev, isOpen: false }))}
      />
    </aside>
  );
}
