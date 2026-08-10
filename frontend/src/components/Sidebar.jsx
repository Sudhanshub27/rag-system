import React, { useState } from 'react';
import { Upload, Trash2, Shield, FileText, Database, Info, AlertTriangle } from 'lucide-react';
import UploadProgress from './UploadProgress';

export default function Sidebar({
  tenantId,
  documents,
  stats,
  onUpload,
  fileProgress,
  onDeleteDoc,
  onDeleteAllData,
}) {
  const [dragActive, setDragActive] = useState(false);

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

  return (
    <aside className="w-80 bg-zinc-950 border-r border-zinc-800/80 flex flex-col h-full text-zinc-300 select-none">
      {/* App Header & Tenant Info */}
      <div className="p-4 border-b border-zinc-800/80 space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm">
            R
          </div>
          <span className="font-semibold text-zinc-100 tracking-tight text-base">Ask My Documents</span>
        </div>
        <div className="text-xs bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 font-mono text-zinc-400 truncate">
          Tenant: <span className="text-indigo-400">{tenantId || 'Loading...'}</span>
        </div>
      </div>

      {/* Main Scrollable Sidebar Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 text-sm">
        {/* Lock / Privacy Note */}
        <div className="flex items-start gap-2.5 p-3 rounded-lg bg-indigo-950/30 border border-indigo-900/40 text-xs text-indigo-300">
          <Shield className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
          <span>
            No login needed. Your documents are stored under an anonymous ID private to this browser.
          </span>
        </div>

        {/* Drag & Drop File Upload */}
        <div className="space-y-2">
          <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Upload Documents</div>
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-colors ${
              dragActive
                ? 'border-indigo-500 bg-indigo-950/20'
                : 'border-zinc-800 hover:border-zinc-700 bg-zinc-900/40'
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
            <label htmlFor="sidebar-file-input" className="cursor-pointer flex flex-col items-center gap-2">
              <Upload className="w-5 h-5 text-zinc-400" />
              <div className="text-xs text-zinc-300">
                <span className="font-medium text-indigo-400">Click to upload</span> or drag and drop
              </div>
              <div className="text-[10px] text-zinc-500">PDF, TXT, MD up to 25MB</div>
            </label>
          </div>
          <UploadProgress fileProgress={fileProgress} />
        </div>

        {/* Knowledge Base Stats */}
        <div className="space-y-2">
          <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Knowledge Base Stats</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded-lg">
              <div className="text-lg font-semibold text-zinc-100">{stats.total_chunks || 0}</div>
              <div className="text-[11px] text-zinc-400 flex items-center gap-1">
                <Database className="w-3 h-3" /> Total Chunks
              </div>
            </div>
            <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded-lg">
              <div className="text-lg font-semibold text-zinc-100">{documents.length || 0}</div>
              <div className="text-[11px] text-zinc-400 flex items-center gap-1">
                <FileText className="w-3 h-3" /> Documents
              </div>
            </div>
          </div>
        </div>

        {/* Ingested Documents List */}
        <div className="space-y-2">
          <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Your Documents</div>
          {documents.length === 0 ? (
            <div className="text-xs text-zinc-500 italic p-2 bg-zinc-900/50 rounded border border-zinc-800/50">
              No files uploaded yet.
            </div>
          ) : (
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {documents.map((doc) => (
                <div
                  key={doc.filename}
                  className="flex items-center justify-between p-2 rounded-lg bg-zinc-900/80 border border-zinc-800/80 text-xs"
                >
                  <div className="truncate max-w-[170px]">
                    <div className="font-medium text-zinc-200 truncate">{doc.filename}</div>
                    <div className="text-[10px] text-zinc-500">{doc.chunk_count} chunks</div>
                  </div>
                  <button
                    onClick={() => onDeleteDoc(doc.filename)}
                    className="p-1 text-zinc-500 hover:text-red-400 transition-colors"
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

      {/* Footer / Purge Data Button */}
      <div className="p-4 border-t border-zinc-800/80 space-y-2">
        <button
          onClick={onDeleteAllData}
          className="w-full py-2 px-3 rounded-lg bg-red-950/40 hover:bg-red-950/80 border border-red-900/50 text-red-300 text-xs font-medium flex items-center justify-center gap-2 transition-colors"
        >
          <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
          Delete all my data
        </button>
      </div>
    </aside>
  );
}
