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
    <div className="border-t border-warmborder bg-parchment-100/90 backdrop-blur-sm p-4 space-y-2">
      {/* Understated Follow-up Links */}
      {followups && followups.length > 0 && !isStreaming && (
        <div className="flex flex-wrap items-center gap-2 max-w-3xl mx-auto px-1 text-xs">
          <span className="text-charcoal-500 font-medium font-sans">Suggested topics:</span>
          {followups.map((f, i) => (
            <button
              key={i}
              onClick={() => onSelectFollowup(f)}
              className="text-terracotta-600 hover:text-terracotta-700 hover:underline font-serif italic text-xs transition-colors"
            >
              "{f}"
            </button>
          ))}
        </div>
      )}

      {/* Input Bar */}
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex items-center gap-2">
        <div className="relative flex-1 flex items-center">
          <BookOpen className="absolute left-3.5 w-4 h-4 text-charcoal-500 pointer-events-none" />
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a research question or request a section breakdown..."
            disabled={isStreaming}
            className="w-full bg-parchment-50 border border-warmborder rounded-xl pl-10 pr-4 py-3 text-sm text-charcoal-900 placeholder-charcoal-500 focus:outline-none focus:border-terracotta-600 focus:ring-1 focus:ring-terracotta-600 transition-all font-sans disabled:opacity-50 shadow-sm"
          />
        </div>
        <button
          type="submit"
          disabled={!input.trim() || isStreaming}
          className="bg-terracotta-600 hover:bg-terracotta-700 disabled:opacity-40 text-parchment-50 rounded-xl p-3 flex items-center justify-center transition-colors shadow-sm"
          title="Submit question"
        >
          <ArrowUp className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
}
