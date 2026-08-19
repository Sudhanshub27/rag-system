import React from 'react';
import {
  BookOpen,
  FileText,
  Scissors,
  Search,
  ShieldCheck,
  CheckCircle2,
  HelpCircle,
  ArrowRight,
  Layers,
  Sparkles,
} from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      num: '01',
      title: 'Upload & Parse Documents',
      desc: 'Upload PDFs, Word (.docx), Excel (.xlsx), CSV, JSON, Markdown, or plain text files. The system extracts text, paragraphs, tables, and sheets while preserving exact page markers and embedded data structures.',
      icon: FileText,
    },
    {
      num: '02',
      title: 'Smart Chunking & Multi-Tenant Container Storage',
      desc: 'Long documents are split into logical passages (250-512 token chunks with sentence boundary preservation). Embeddings are indexed into an isolated, per-tenant ChromaDB vector database container (tenant_<id>) so your data is structurally isolated from other users.',
      icon: Scissors,
    },
    {
      num: '03',
      title: 'Dual Query Routing: Narrow vs. Broad Intent',
      desc: 'An automated router (QueryRouter) analyzes your question. Specific lookup queries execute precision Hybrid Search (BM25 + Dense Vector RRF fusion + Cross-Encoder reranking). Broad questions ("summarize this document") execute Map-Reduce summarization across all chunks and cache the result keyed by tenant and document SHA-256 hash.',
      icon: Search,
    },
    {
      num: '04',
      title: 'Local Privacy & PII Scrubbing',
      desc: 'Personal details (names, emails, phone numbers, IP addresses) are redacted locally on your machine before excerpts are processed by LLM APIs, ensuring zero personal data exposure to third-party endpoints.',
      icon: ShieldCheck,
    },
    {
      num: '05',
      title: 'Grounded Answer with Page Citations',
      desc: 'The AI generates a clear answer strictly grounded in your text. Every claim includes a clickable inline citation pointing directly to the source page and excerpt in the split-screen inspector.',
      icon: CheckCircle2,
    },
  ];

  return (
    <div className="w-full h-full overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans px-4 sm:px-8 py-6 sm:py-10">
      <div className="max-w-3xl mx-auto space-y-8 sm:space-y-12 pb-16">
        {/* Header */}
        <div className="border-b border-warmborder pb-5 sm:pb-6 space-y-2">
          <div className="flex items-center gap-2 text-terracotta-600 font-semibold text-xs uppercase tracking-wider font-sans">
            <BookOpen className="w-4 h-4 shrink-0" /> Operational Guide
          </div>
          <h1 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight">
            How It Works
          </h1>
          <p className="font-serif italic text-charcoal-700 text-sm sm:text-base">
            From document upload to cited, verifiable answers — step by step.
          </p>
        </div>

        {/* Step-by-Step Explanation Section */}
        <div className="space-y-6">
          <div className="text-xs uppercase font-semibold tracking-wider text-charcoal-500 font-sans flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-terracotta-600 shrink-0" /> The 5-Step Process
          </div>

          <div className="space-y-4">
            {steps.map((step, idx) => {
              const Icon = step.icon;
              return (
                <div
                  key={idx}
                  className="p-4 sm:p-5 rounded-xl bg-parchment-50 border border-warmborder shadow-2xs flex items-start gap-3.5 sm:gap-4 transition-all hover:border-terracotta-600/40"
                >
                  <div className="flex flex-col items-center justify-center shrink-0">
                    <span className="font-serif font-bold text-base sm:text-lg text-terracotta-600">
                      {step.num}
                    </span>
                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-terracotta-100/60 border border-terracotta-600/20 flex items-center justify-center mt-1">
                      <Icon className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-terracotta-600" />
                    </div>
                  </div>
                  <div className="space-y-1 flex-1 min-w-0">
                    <h3 className="font-serif font-bold text-base sm:text-lg text-charcoal-900 flex items-center justify-between">
                      <span>{step.title}</span>
                    </h3>
                    <p className="text-xs sm:text-sm text-charcoal-700 leading-relaxed font-sans">
                      {step.desc}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* SEPARATE SECTION: Standard Chatbots Comparison */}
        <div className="border-t border-warmborder pt-8 space-y-6">
          <div className="space-y-2">
            <div className="text-xs uppercase font-semibold tracking-wider text-charcoal-500 font-sans flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-terracotta-600 shrink-0" /> Comparison Guide
            </div>
            <h2 className="font-serif font-bold text-xl sm:text-2xl text-charcoal-900 tracking-tight">
              Why use this instead of pasting PDFs into ChatGPT or Claude?
            </h2>
            <p className="text-xs sm:text-sm text-charcoal-700 font-serif italic">
              General conversational chat windows are great for writing assistance, but they have major limitations when querying complex documents.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm font-sans">
            <div className="p-4 sm:p-5 rounded-xl bg-parchment-50 border border-warmborder shadow-2xs space-y-2">
              <h3 className="font-serif font-bold text-sm sm:text-base text-charcoal-900 flex items-center gap-2">
                <FileText className="w-4 h-4 text-terracotta-600 shrink-0" />
                No Silent Document Truncation
              </h3>
              <p className="text-xs text-charcoal-700 leading-relaxed">
                Pasting long PDFs into standard chat boxes often quietly cuts off pages past token limits. Here, 100% of your document is indexed and fully searchable.
              </p>
            </div>

            <div className="p-4 sm:p-5 rounded-xl bg-parchment-50 border border-warmborder shadow-2xs space-y-2">
              <h3 className="font-serif font-bold text-sm sm:text-base text-charcoal-900 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-terracotta-600 shrink-0" />
                Clickable Source Page Citations
              </h3>
              <p className="text-xs text-charcoal-700 leading-relaxed">
                Standard chatbots ask you to trust their answers on faith. Every claim generated here carries a clickable citation pointing directly to the exact page excerpt.
              </p>
            </div>

            <div className="p-4 sm:p-5 rounded-xl bg-parchment-50 border border-warmborder shadow-2xs space-y-2">
              <h3 className="font-serif font-bold text-sm sm:text-base text-charcoal-900 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-terracotta-600 shrink-0" />
                Zero Hallucinations Guarantee
              </h3>
              <p className="text-xs text-charcoal-700 leading-relaxed">
                General chat models fill gaps by guessing from general web training. This system strictly uses your uploaded text; if the answer isn't in your document, it explicitly tells you.
              </p>
            </div>

            <div className="p-4 sm:p-5 rounded-xl bg-parchment-50 border border-warmborder shadow-2xs space-y-2">
              <h3 className="font-serif font-bold text-sm sm:text-base text-charcoal-900 flex items-center gap-2">
                <Layers className="w-4 h-4 text-terracotta-600 shrink-0" />
                Persistent Multi-File Knowledge
              </h3>
              <p className="text-xs text-charcoal-700 leading-relaxed">
                Standard chat windows lose context when closed. Upload your files once here, and return anytime to ask new questions across multiple documents simultaneously.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
