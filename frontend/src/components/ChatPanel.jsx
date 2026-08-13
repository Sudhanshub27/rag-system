import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Copy, Check, Download, AlertCircle, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { markdownToPlainText } from '../utils/format';

export default function ChatPanel({
  messages,
  onSendMessage,
  isStreaming,
  currentStreamText,
  currentCitations,
  onSelectCitation,
  error,
  followups,
  onSelectFollowup,
}) {
  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentStreamText]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleCopy = (text, id) => {
    const plainText = markdownToPlainText(text);
    navigator.clipboard.writeText(plainText);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleExportMarkdown = (text, idx) => {
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `answer_${idx + 1}.md`;
    a.click();
  };

  return (
    <main className="flex-1 flex flex-col h-full bg-zinc-900 text-zinc-100 overflow-hidden">
      {/* Header */}
      <header className="p-4 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/60">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-indigo-400" />
          <h1 className="font-semibold text-zinc-100 text-sm">RAG Assistant Thread</h1>
        </div>
      </header>

      {/* Error Alert Banner */}
      {error && (
        <div className="bg-red-950/60 border-b border-red-900/60 p-3 px-4 flex items-center gap-2 text-xs text-red-300">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Messages Thread */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 && !isStreaming ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
            <div className="w-12 h-12 rounded-full bg-indigo-950/50 border border-indigo-900/50 flex items-center justify-center text-indigo-400">
              <Sparkles className="w-6 h-6" />
            </div>
            <h3 className="text-base font-semibold text-zinc-200">Ask questions about your documents</h3>
            <p className="text-xs text-zinc-500 max-w-sm">
              Upload PDFs or text files on the sidebar, then ask questions or request summaries.
            </p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} space-y-1.5`}>
              <div className="text-[10px] text-zinc-500 font-medium px-1">
                {msg.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div
                className={`p-4 rounded-2xl max-w-3xl text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white rounded-tr-none'
                    : 'bg-zinc-950 border border-zinc-800 rounded-tl-none text-zinc-200 space-y-3'
                }`}
              >
                {msg.role === 'user' ? (
                  <div>{msg.content}</div>
                ) : (
                  <>
                    <div className="prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    </div>

                    {/* Citations Badges */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="pt-2 border-t border-zinc-800/80 flex flex-wrap gap-2 text-xs">
                        <span className="text-[11px] text-zinc-400 font-medium self-center">Sources:</span>
                        {msg.citations.map((c) => (
                          <button
                            key={c.id}
                            onClick={() => onSelectCitation(c)}
                            className="px-2 py-0.5 rounded bg-indigo-950/50 border border-indigo-800/60 text-indigo-300 hover:bg-indigo-900/60 font-mono text-[11px] flex items-center gap-1 transition-colors"
                          >
                            <FileText className="w-3 h-3" />
                            {c.source} (p.{c.page})
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Actions: Copy & Export */}
                    <div className="flex items-center gap-2 pt-1">
                      <button
                        onClick={() => handleCopy(msg.content, idx)}
                        className="text-zinc-500 hover:text-zinc-300 text-xs flex items-center gap-1"
                      >
                        {copiedId === idx ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                        {copiedId === idx ? 'Copied' : 'Copy'}
                      </button>
                      <button
                        onClick={() => handleExportMarkdown(msg.content, idx)}
                        className="text-zinc-500 hover:text-zinc-300 text-xs flex items-center gap-1"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Markdown
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          ))
        )}

        {/* Live Streaming Response */}
        {isStreaming && (
          <div className="flex flex-col items-start space-y-1.5">
            <div className="text-[10px] text-zinc-500 font-medium px-1">Assistant</div>
            <div className="p-4 rounded-2xl rounded-tl-none bg-zinc-950 border border-zinc-800 text-zinc-200 text-sm max-w-3xl leading-relaxed space-y-3">
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{currentStreamText}</ReactMarkdown>
                <span className="inline-block w-2 h-4 bg-indigo-400 animate-pulse ml-1 align-middle" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Contextual Follow-up Chips */}
      {followups && followups.length > 0 && !isStreaming && (
        <div className="px-4 py-2 bg-zinc-950/40 border-t border-zinc-800/60 flex flex-wrap gap-2">
          <span className="text-[11px] text-zinc-500 self-center">Suggested follow-ups:</span>
          {followups.map((f, i) => (
            <button
              key={i}
              onClick={() => onSelectFollowup(f)}
              className="text-xs px-2.5 py-1 rounded-full bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700 transition-colors"
            >
              {f}
            </button>
          ))}
        </div>
      )}

      {/* Input Box */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-zinc-800 bg-zinc-950/60 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question or request a summary..."
          disabled={isStreaming}
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-indigo-500 transition-colors disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || isStreaming}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl px-4 py-2.5 text-sm font-medium flex items-center gap-1.5 transition-colors"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </main>
  );
}
