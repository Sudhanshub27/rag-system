import React from 'react';
import { FileText, X, Bookmark } from 'lucide-react';

export default function SourceInspector({ selectedCitation, onClose }) {
  if (!selectedCitation) {
    return (
      <aside className="w-80 bg-parchment-200 border-l border-warmborder flex flex-col h-full text-charcoal-500 select-none font-sans">
        <div className="p-4 border-b border-warmborder font-serif font-bold text-charcoal-900 text-sm flex items-center justify-between">
          <span>Source Inspector</span>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-xs space-y-3">
          <FileText className="w-8 h-8 text-charcoal-500" />
          <p className="font-serif italic text-charcoal-700">
            Hover over or click any citation in the manuscript to inspect original page text and metadata.
          </p>
        </div>
      </aside>
    );
  }

  const { source, page, text, chunks, score, id } = selectedCitation;
  const displayTextChunks = chunks && chunks.length > 0 ? chunks : text ? [text] : [];

  return (
    <aside className="w-80 bg-parchment-200 border-l border-warmborder flex flex-col h-full text-charcoal-900 font-sans shadow-inner">
      {/* Inspector Header */}
      <div className="p-4 border-b border-warmborder font-serif font-bold text-charcoal-900 text-sm flex items-center justify-between">
        <span className="flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-terracotta-600" />
          Source Entry [{id}]
        </span>
        <button onClick={onClose} className="text-charcoal-500 hover:text-charcoal-900 transition-colors">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
        {/* Source File Card */}
        <div className="bg-parchment-50 border border-warmborder rounded-lg p-3 space-y-2 shadow-sm">
          <div className="text-[10px] uppercase font-semibold text-charcoal-500 tracking-wider">Document Page</div>
          <div className="font-serif font-bold text-charcoal-900 text-sm truncate flex items-center gap-2">
            <FileText className="w-4 h-4 text-terracotta-600 shrink-0" />
            <span className="truncate">{source}</span>
          </div>
          <div className="flex justify-between items-center text-charcoal-700 text-xs pt-2 border-t border-warmborder font-serif">
            <span>Page Number: <strong className="text-charcoal-900 font-sans font-semibold">{page}</strong></span>
            {score !== undefined && (
              <span className="text-sage-600 font-sans font-semibold">
                Relevance: {(score * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>

        {/* Page Content / Chunks */}
        <div className="space-y-2">
          <div className="text-[10px] uppercase font-semibold text-charcoal-500 tracking-wider">Page Excerpt</div>
          <div className="space-y-2">
            {displayTextChunks.map((chunkText, idx) => (
              <div
                key={idx}
                className="bg-parchment-50 border border-warmborder rounded-lg p-3 text-charcoal-900 leading-relaxed font-serif text-xs whitespace-pre-wrap shadow-sm"
              >
                {chunkText}
              </div>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}
