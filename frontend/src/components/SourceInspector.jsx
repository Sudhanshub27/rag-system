import React from 'react';
import { FileText, X, Bookmark } from 'lucide-react';

export default function SourceInspector({ selectedCitation, onClose, debugScores, isMobileDrawer = false }) {
  if (!selectedCitation) {
    return (
      <aside className={`${isMobileDrawer ? 'w-full h-full' : 'hidden md:flex w-80 border-l'} shrink-0 bg-parchment-200 border-warmborder flex flex-col h-full text-charcoal-500 select-none font-sans overflow-hidden`}>
        <div className="p-4 border-b border-warmborder font-serif font-bold text-charcoal-900 text-sm flex items-center justify-between bg-parchment-200/90 shrink-0">
          <span className="flex items-center gap-2 font-sans font-semibold text-charcoal-700">
            <Bookmark className="w-4 h-4 text-terracotta-600" /> Source Inspector
          </span>
          {isMobileDrawer && (
            <button
              onClick={onClose}
              className="flex items-center justify-center min-w-[44px] min-h-[44px] p-2 text-charcoal-500 hover:text-charcoal-900 transition-colors rounded-xl hover:bg-parchment-50 cursor-pointer"
              title="Close Inspector"
              aria-label="Close Inspector"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-xs space-y-4">
          <div className="w-12 h-12 rounded-full bg-parchment-50 border border-warmborder flex items-center justify-center text-terracotta-600 shadow-2xs">
            <FileText className="w-6 h-6" />
          </div>
          <p className="font-serif italic text-charcoal-700 leading-relaxed text-xs max-w-[200px]">
            Click any inline citation marker <span className="text-terracotta-600 font-mono font-bold font-sans">[n]</span> or citation badge to inspect original page text and metadata.
          </p>
        </div>
      </aside>
    );
  }

  const rawSource = selectedCitation.source || selectedCitation.fullPath || 'Document';
  const filename = rawSource.split('/').pop();
  const page = selectedCitation.page !== undefined && selectedCitation.page !== null ? selectedCitation.page : 1;
  const citationId = selectedCitation.id || 1;
  const score = selectedCitation.score;

  // Extract chunks or text
  const displayTextChunks = selectedCitation.chunks && selectedCitation.chunks.length > 0
    ? selectedCitation.chunks
    : selectedCitation.text
    ? [selectedCitation.text]
    : [];

  return (
    <aside className={`${isMobileDrawer ? 'w-full h-full' : 'hidden md:flex w-80 border-l'} shrink-0 bg-parchment-200 border-warmborder flex flex-col h-full text-charcoal-900 font-sans shadow-inner overflow-hidden animate-fadeIn`}>
      {/* Inspector Header */}
      <div className="p-4 border-b border-warmborder font-serif font-bold text-charcoal-900 text-sm flex items-center justify-between bg-parchment-200/90 shrink-0">
        <span className="flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-terracotta-600" />
          Citation Source [{citationId}]
        </span>
        <button
          onClick={onClose}
          className="flex items-center justify-center min-w-[44px] min-h-[44px] p-2 text-charcoal-500 hover:text-charcoal-900 transition-colors rounded-xl hover:bg-parchment-50 cursor-pointer"
          title="Close Inspector"
          aria-label="Close Inspector"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
        {/* Source File Metadata Card */}
        <div className="bg-parchment-50 border border-warmborder rounded-lg p-3.5 space-y-2 shadow-2xs">
          <div className="text-[10px] uppercase font-bold text-charcoal-700 tracking-wider font-sans">Document Metadata</div>
          <div className="font-serif font-bold text-charcoal-900 text-sm truncate flex items-center gap-2" title={filename}>
            <FileText className="w-4 h-4 text-terracotta-700 shrink-0" />
            <span className="truncate">{filename}</span>
          </div>
          <div className="flex justify-between items-center text-charcoal-800 text-xs pt-2 border-t border-warmborder font-serif">
            <span>Page Number: <strong className="text-charcoal-900 font-sans font-semibold">{page}</strong></span>
            {score !== undefined && debugScores && (
              <span className="text-sage-700 font-sans font-semibold font-mono">
                {(score * 100).toFixed(0)}% Relevance
              </span>
            )}
          </div>
        </div>

        {/* Page Excerpt Text */}
        <div className="space-y-2">
          <div className="text-[10px] uppercase font-bold text-charcoal-700 tracking-wider font-sans">
            Original Document Excerpt
          </div>
          {displayTextChunks.length === 0 ? (
            <div className="bg-parchment-50 border border-warmborder rounded-lg p-3 text-charcoal-700 italic text-xs">
              No excerpt text recorded for this chunk.
            </div>
          ) : (
            displayTextChunks.map((chunkText, idx) => (
              <div
                key={idx}
                className="bg-parchment-50 border border-warmborder rounded-lg p-3.5 text-charcoal-900 leading-relaxed font-serif text-xs whitespace-pre-wrap shadow-2xs space-y-1"
              >
                <div className="text-[9px] font-mono text-terracotta-700 uppercase tracking-wider font-bold mb-1">
                  Excerpt #{idx + 1}
                </div>
                <div>{chunkText}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </aside>
  );
}
