import React from 'react';
import { CheckCircle2, Loader2, AlertCircle } from 'lucide-react';

export default function UploadProgress({ fileProgress }) {
  if (!fileProgress || Object.keys(fileProgress).length === 0) return null;

  const stages = [
    { id: 'chunking', label: 'Chunking' },
    { id: 'embedding', label: 'Embedding' },
    { id: 'indexing', label: 'Indexing' },
    { id: 'complete', label: 'Complete' },
  ];

  return (
    <div className="space-y-3 p-3 bg-zinc-900 border border-zinc-800 rounded-lg text-xs">
      <div className="font-semibold text-zinc-300">Ingestion Progress</div>
      {Object.entries(fileProgress).map(([filename, prog]) => {
        const isError = prog.stage === 'error';
        const isComplete = prog.stage === 'complete';

        return (
          <div key={filename} className="space-y-1.5 border-b border-zinc-800/80 pb-2 last:border-0 last:pb-0">
            <div className="flex items-center justify-between text-zinc-400">
              <span className="font-medium truncate max-w-[160px] text-zinc-200">{filename}</span>
              <span className="text-[10px] uppercase font-mono">
                {isError ? (
                  <span className="text-red-400 flex items-center gap-1"><AlertCircle className="w-3 h-3" /> Error</span>
                ) : isComplete ? (
                  <span className="text-emerald-400 flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Done ({prog.chunks_added} chunks)</span>
                ) : (
                  <span className="text-indigo-400 flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> {prog.stage}</span>
                )}
              </span>
            </div>

            {/* Progress Bar */}
            <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-1.5 transition-all duration-300 ${
                  isError ? 'bg-red-500' : isComplete ? 'bg-emerald-500' : 'bg-indigo-500'
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
