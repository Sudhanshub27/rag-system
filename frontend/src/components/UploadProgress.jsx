import React from 'react';
import { CheckCircle2, Loader2, AlertCircle } from 'lucide-react';

export default function UploadProgress({ fileProgress }) {
  if (!fileProgress || Object.keys(fileProgress).length === 0) return null;

  return (
    <div className="space-y-3 p-3 bg-parchment-50 border border-warmborder rounded-lg text-xs font-sans shadow-sm">
      <div className="font-semibold text-charcoal-700">Ingestion Status</div>
      {Object.entries(fileProgress).map(([filename, prog]) => {
        const isError = prog.stage === 'error';
        const isComplete = prog.stage === 'complete';

        return (
          <div key={filename} className="space-y-1.5 border-b border-warmborder pb-2 last:border-0 last:pb-0">
            <div className="flex items-center justify-between text-charcoal-700">
              <span className="font-serif font-medium truncate max-w-[150px] text-charcoal-900">{filename}</span>
              <span className="text-[10px] uppercase font-mono">
                {isError ? (
                  <span className="text-rust-600 flex items-center gap-1"><AlertCircle className="w-3 h-3" /> Failed</span>
                ) : isComplete ? (
                  <span className="text-sage-600 flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Ready ({prog.chunks_added} chunks)</span>
                ) : (
                  <span className="text-terracotta-600 flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> {prog.stage}</span>
                )}
              </span>
            </div>

            <div className="w-full bg-parchment-200 rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-1.5 transition-all duration-300 ${
                  isError ? 'bg-rust-600' : isComplete ? 'bg-sage-600' : 'bg-terracotta-600'
                }`}
                style={{ width: `${prog.progress || 0}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
