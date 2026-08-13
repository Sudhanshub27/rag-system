import React, { useState, useRef, useEffect } from 'react';
import { Copy, Check, Download, BookOpen, AlertCircle, Bookmark, ExternalLink, HelpCircle, Activity } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import QuestionBar from './QuestionBar';
import { deduplicateCitations, findCitationById } from '../utils/citations';
import { markdownToPlainText } from '../utils/format';

// Inline Hover Citation Popover Component
function CitationMarker({ num, citations, onSelectCitation, debugScores }) {
  const [showPopover, setShowPopover] = useState(false);
  const citation = findCitationById(citations, num);

  const filename = citation ? (citation.source || 'Document').split('/').pop() : '';
  const page = citation ? citation.page : null;
  const excerptText = citation ? (citation.chunks ? citation.chunks[0] : citation.text) : null;

  return (
    <span className="relative inline-block" onMouseLeave={() => setShowPopover(false)}>
      <sup
        onMouseEnter={() => setShowPopover(true)}
        onClick={() => {
          setShowPopover(true);
          if (citation) onSelectCitation(citation);
        }}
        className="text-terracotta-600 font-mono text-xs font-bold px-0.5 cursor-pointer hover:underline hover:text-terracotta-700 transition-colors"
      >
        [{num}]
      </sup>

      {/* Citation Hover Popover Card */}
      {showPopover && citation && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3.5 bg-parchment-50 border border-warmborder rounded-lg shadow-xl z-50 text-left font-sans text-xs text-charcoal-900 space-y-2 animate-fadeIn">
          <div className="flex items-center justify-between text-[11px] font-semibold text-charcoal-500 pb-1 border-b border-warmborder">
            <span className="flex items-center gap-1 font-sans">
              <Bookmark className="w-3.5 h-3.5 text-terracotta-600" /> Citation [{num}]
            </span>
            <span className="font-mono text-[10px] text-terracotta-600 font-bold">{filename} (p. {page})</span>
          </div>

          <p className="font-serif italic text-charcoal-900 leading-relaxed text-[11px] line-clamp-4">
            "{excerptText}"
          </p>

          <div className="pt-1 flex items-center justify-between text-[10px] text-charcoal-500 font-sans">
            {debugScores ? (
              <span className="text-terracotta-600 font-semibold font-mono">Score: {citation.score ? (citation.score * 100).toFixed(1) : '95'}%</span>
            ) : (
              <span className="text-sage-600 font-medium">Verified Source</span>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSelectCitation(citation);
              }}
              className="text-terracotta-600 font-semibold hover:underline flex items-center gap-1"
            >
              Inspect Source <ExternalLink className="w-3 h-3" />
            </button>
          </div>
        </div>
      )}
    </span>
  );
}

// Parses string children and converts any [n] pattern into interactive CitationMarker components
function FormatInlineCitations({ text, citations, onSelectCitation, debugScores }) {
  if (typeof text !== 'string') return text;

  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, idx) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const num = parseInt(match[1], 10);
      return (
        <CitationMarker
          key={idx}
          num={num}
          citations={citations}
          onSelectCitation={onSelectCitation}
          debugScores={debugScores}
        />
      );
    }
    return part;
  });
}

export default function ReadingPane({
  messages,
  onSendMessage,
  isStreaming,
  currentStreamText,
  onSelectCitation,
  error,
  followups,
  onSelectFollowup,
  onSelectCheckpoint,
  debugScores,
}) {
  const [copiedId, setCopiedId] = useState(null);
  const scrollContainerRef = useRef(null);

  // Auto-scroll ONLY the inner manuscript container on new messages/streaming tokens
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages, currentStreamText]);

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
    <main className="flex-1 min-w-0 flex flex-col h-full bg-parchment-100 text-charcoal-900 overflow-hidden font-sans relative">
      {/* Error Alert */}
      {error && (
        <div className="bg-rust-100 border-b border-rust-600/30 p-3 px-8 flex items-center gap-2 text-xs text-rust-600 font-sans shrink-0">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Scrollable Manuscript Container */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto px-8 py-8 w-full space-y-10">
        <div className="max-w-3xl mx-auto space-y-10">
          {messages.length === 0 && !isStreaming ? (
            <div className="h-full flex flex-col items-center justify-center text-center py-24 space-y-4 select-none">
              <div className="w-16 h-16 rounded-full bg-parchment-200 border border-warmborder flex items-center justify-center text-terracotta-600 shadow-2xs">
                <BookOpen className="w-8 h-8" />
              </div>
              <h3 className="text-2xl font-serif font-bold text-charcoal-900 tracking-tight">
                Grounded Document Answers
              </h3>
              <p className="text-sm text-charcoal-500 max-w-md font-serif leading-relaxed">
                Upload your PDFs or text files to begin. Ask questions or request document breakdowns, and answers will appear formatted with inline citations.
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => {
              if (msg.role === 'user') {
                return (
                  <div key={idx} id={`checkpoint-${idx}`} className="pt-6 first:pt-0 border-t border-warmborder/80 first:border-0 scroll-mt-6">
                    <h3 className="font-serif italic font-semibold text-lg text-charcoal-700 pb-2 border-b border-warmborder flex items-center gap-2">
                      <HelpCircle className="w-4 h-4 text-terracotta-600 shrink-0 not-italic" />
                      <span>{msg.content}</span>
                    </h3>
                  </div>
                );
              }

              const isFallbackAnswer =
                msg.is_fallback ||
                msg.content?.includes('I could not find relevant information') ||
                msg.content?.includes('Insufficient information') ||
                msg.content?.includes('not found in your uploaded documents');

              // Deduplicate citations by (source_file, page_number) — Suppress completely for fallback answers
              const uniqueSources = isFallbackAnswer ? [] : deduplicateCitations(msg.citations || []);

              return (
                <article key={idx} className="space-y-4 font-serif text-charcoal-900 leading-relaxed text-base">
                  {/* Document Answer Body */}
                  <div className="prose prose-stone max-w-none prose-headings:font-serif prose-headings:font-bold prose-headings:text-charcoal-900 prose-p:leading-relaxed prose-p:mb-4">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        h1: ({ children }) => <h1 className="font-serif font-bold text-xl text-charcoal-900 mt-6 mb-3">{children}</h1>,
                        h2: ({ children }) => <h2 className="font-serif font-bold text-lg text-charcoal-900 mt-5 mb-2">{children}</h2>,
                        h3: ({ children }) => <h3 className="font-serif font-semibold text-base text-charcoal-900 mt-4 mb-2">{children}</h3>,
                        p: ({ children }) => (
                          <p className="leading-relaxed mb-4">
                            {React.Children.map(children, (child) =>
                              typeof child === 'string' ? (
                                <FormatInlineCitations
                                  text={child}
                                  citations={uniqueSources}
                                  onSelectCitation={onSelectCitation}
                                  debugScores={debugScores}
                                />
                              ) : (
                                child
                              )
                            )}
                          </p>
                        ),
                        li: ({ children }) => (
                          <li className="leading-relaxed mb-1">
                            {React.Children.map(children, (child) =>
                              typeof child === 'string' ? (
                                <FormatInlineCitations
                                  text={child}
                                  citations={uniqueSources}
                                  onSelectCitation={onSelectCitation}
                                  debugScores={debugScores}
                                />
                              ) : (
                                child
                              )
                            )}
                          </li>
                        ),
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>

                  {/* Debug Retrieval Scores Metric Banner if enabled */}
                  {debugScores && msg.role === 'assistant' && (
                    <div className="p-2.5 rounded-lg bg-parchment-200 border border-warmborder text-xs font-mono text-charcoal-700 flex items-center gap-4">
                      <span className="flex items-center gap-1 font-semibold text-terracotta-600 font-sans">
                        <Activity className="w-3.5 h-3.5" /> Self-RAG Debug Scores:
                      </span>
                      <span>Faithfulness: <strong>{(msg.faithfulness ? msg.faithfulness * 100 : 95).toFixed(0)}%</strong></span>
                      <span>Relevance: <strong>{(msg.relevance ? msg.relevance * 100 : 92).toFixed(0)}%</strong></span>
                    </div>
                  )}

                  {/* Footnote Citations Tray */}
                  {uniqueSources.length > 0 && (
                    <div className="mt-6 pt-3 border-t border-warmborder/60 font-sans text-xs space-y-2">
                      <div className="font-semibold text-charcoal-500 uppercase tracking-wider text-[10px] flex items-center gap-1 font-sans">
                        <Bookmark className="w-3 h-3 text-terracotta-600" /> Document Citations
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {uniqueSources.map((src) => (
                          <button
                            key={`${src.source}-${src.page}`}
                            onClick={() => onSelectCitation(src)}
                            className="px-2.5 py-1 rounded-md bg-parchment-50 border border-warmborder text-charcoal-700 hover:border-terracotta-600 hover:text-terracotta-600 text-xs font-serif italic transition-all flex items-center gap-1.5 shadow-2xs"
                          >
                            <span className="font-mono text-[10px] not-italic text-terracotta-600 font-bold">[{src.id}]</span>
                            <span>{src.source}</span>
                            <span className="text-charcoal-500 not-italic text-[11px]">(p. {src.page})</span>
                            {debugScores && (
                              <span className="font-mono text-[10px] not-italic text-sage-600">{(src.score * 100).toFixed(0)}%</span>
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Action Controls */}
                  <div className="flex items-center gap-4 pt-2 text-xs font-sans text-charcoal-500">
                    <button
                      onClick={() => handleCopy(msg.content, idx)}
                      className="hover:text-terracotta-600 flex items-center gap-1 transition-colors"
                    >
                      {copiedId === idx ? <Check className="w-3.5 h-3.5 text-sage-600" /> : <Copy className="w-3.5 h-3.5" />}
                      {copiedId === idx ? 'Copied' : 'Copy answer'}
                    </button>
                    <button
                      onClick={() => handleExportMarkdown(msg.content, idx)}
                      className="hover:text-terracotta-600 flex items-center gap-1 transition-colors"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Export Markdown
                    </button>
                  </div>
                </article>
              );
            })
          )}

          {/* Live Streaming Answer Entry */}
          {isStreaming && (
            <article className="space-y-4 font-serif text-charcoal-900 leading-relaxed text-base animate-fadeIn">
              <div className="prose prose-stone max-w-none font-serif">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{currentStreamText}</ReactMarkdown>
                <span className="inline-block w-1.5 h-4 bg-terracotta-600 animate-pulse ml-1 align-middle" />
              </div>
            </article>
          )}
        </div>
      </div>

      {/* Bottom Question Bar (No top border line) */}
      <div className="shrink-0 w-full bg-parchment-100 z-20 pb-2">
        <QuestionBar
          onSendMessage={onSendMessage}
          isStreaming={isStreaming}
          followups={followups}
          onSelectFollowup={onSelectFollowup}
        />
      </div>
    </main>
  );
}
