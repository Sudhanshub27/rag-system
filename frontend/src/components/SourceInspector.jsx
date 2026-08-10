import React from 'react';
import { FileText, X, Bookmark, ExternalLink } from 'lucide-react';

export default function SourceInspector({ selectedCitation, onClose }) {
  if (!selectedCitation) {
    return (
      <aside className="w-80 bg-zinc-950 border-l border-zinc-800/80 flex flex-col h-full text-zinc-400 select-none">
        <div className="p-4 border-b border-zinc-800/80 font-semibold text-zinc-200 text-sm flex items-center justify-between">
          <span>Source Inspector</span>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-xs space-y-3">
          <FileText className="w-8 h-8 text-zinc-600" />
          <p className="text-zinc-500">
            Click on any citation badge <span className="text-indigo-400 font-mono">[1]</span> in the answer to view its original source text and page details.
          </p>
        </div>
      </aside>
    );
  }

  const { source, page, text, score, id } = selectedCitation;

  return (
    <aside className="w-80 bg-zinc-950 border-l border-zinc-800/80 flex flex-col h-full text-zinc-300">
      <div className="p-4 border-b border-zinc-800/80 font-semibold text-zinc-200 text-sm flex items-center justify-between">
        <span className="flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-indigo-400" />
          Citation [{id}]
        </span>
        <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
        {/* Source File & Page Meta */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 space-y-1.5">
          <div className="text-[10px] uppercase font-semibold text-zinc-500 tracking-wider">Document Source</div>
          <div className="font-medium text-zinc-200 truncate flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <span className="truncate">{source}</span>
          </div>
          <div className="flex justify-between items-center text-zinc-400 text-[11px] pt-1 border-t border-zinc-800/60">
            <span>Page Number: <strong className="text-zinc-200">{page}</strong></span>
            {score !== undefined && (
              <span>Relevance: <strong className="text-emerald-400">{(score * 100).toFixed(0)}%</strong></span>
            )}
          </div>
        </div>

        {/* Text Excerpt */}
        <div className="space-y-1.5">
          <div className="text-[10px] uppercase font-semibold text-zinc-500 tracking-wider">Extracted Text Chunk</div>
          <div className="bg-zinc-900/90 border border-zinc-800/80 rounded-lg p-3 text-zinc-300 leading-relaxed font-mono text-[11px] whitespace-pre-wrap">
            {text}
          </div>
        </div>
      </div>
    </aside>
  );
}
