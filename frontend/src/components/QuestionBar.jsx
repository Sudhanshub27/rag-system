import React, { useState } from 'react';
import { ArrowUp, BookOpen } from 'lucide-react';

export default function QuestionBar({ onSendMessage, isStreaming, followups, onSelectFollowup }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    onSendMessage(input.trim());
    setInput('');
  };

  return (
    <div className="bg-parchment-100 p-3 sm:p-4 space-y-2 select-none border-t border-warmborder/60">
      {/* Understated Follow-up Suggestion Pills */}
      {followups && followups.length > 0 && !isStreaming && (
        <div className="flex flex-wrap items-center gap-1.5 max-w-3xl mx-auto px-1 text-xs">
          <span className="text-charcoal-700 font-bold font-sans text-[11px]">Suggested topics:</span>
          {followups.map((f, i) => (
            <button
              key={i}
              type="button"
              onClick={() => onSelectFollowup(f)}
              aria-label={`Ask suggested topic: ${f}`}
              className="px-3 py-1.5 min-h-[36px] rounded-full bg-parchment-50 border border-warmborder text-terracotta-700 hover:text-terracotta-800 hover:border-terracotta-600/50 font-serif italic text-xs transition-colors shadow-2xs cursor-pointer flex items-center gap-1"
            >
              <span>"{f}"</span>
            </button>
          ))}
        </div>
      )}

      {/* Input Bar */}
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex items-center gap-2 w-full min-w-0">
        <div className="relative flex-1 flex items-center min-w-0">
          <BookOpen className="absolute left-3.5 w-4 h-4 text-charcoal-700 pointer-events-none shrink-0" />
          <input
            id="question-bar-input"
            aria-label="Ask a research question or request a section breakdown"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a research question or request a section breakdown..."
            disabled={isStreaming}
            className="w-full bg-parchment-50 border border-warmborder rounded-xl pl-10 pr-4 py-3 min-h-[44px] text-sm text-charcoal-900 placeholder:text-charcoal-600 focus:outline-none focus:border-terracotta-600 focus:ring-1 focus:ring-terracotta-600 transition-all font-sans disabled:opacity-50 shadow-2xs min-w-0"
          />
        </div>
        <button
          type="submit"
          disabled={!input.trim() || isStreaming}
          className="bg-terracotta-600 hover:bg-terracotta-700 disabled:opacity-40 text-parchment-50 rounded-xl min-w-[44px] min-h-[44px] p-2.5 flex items-center justify-center transition-colors shadow-2xs shrink-0 cursor-pointer"
          title="Submit question"
          aria-label="Submit question"
        >
          <ArrowUp className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
}
