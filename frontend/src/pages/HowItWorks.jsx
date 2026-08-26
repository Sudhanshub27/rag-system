import React from 'react';
import { Link } from 'react-router-dom';
import {
  BookOpen,
  FileText,
  Scissors,
  Search,
  ShieldCheck,
  CheckCircle2,
  ArrowRight,
  Layers,
  Sparkles,
  ArrowLeft,
  Cpu,
  Bookmark,
  Database,
  Lock,
} from 'lucide-react';
import LandingFooter from '../components/landing/LandingFooter';

export default function HowItWorks() {
  const steps = [
    {
      num: '01',
      title: 'Upload & Parse Documents',
      desc: 'Upload PDFs, Word (.docx), Excel (.xlsx), CSV, JSON, Markdown, or plain text files. The system extracts text, paragraphs, tables, and sheets while preserving exact page markers and embedded data structures.',
      icon: FileText,
      badge: 'Multi-Format Ingestion',
    },
    {
      num: '02',
      title: 'Smart Chunking & Isolated Tenant Storage',
      desc: 'Long documents are split into logical passages (250-512 token chunks with sentence boundary preservation). Embeddings are indexed into an isolated, per-tenant ChromaDB vector database container (tenant_<id>) so your data is structurally isolated from other users.',
      icon: Scissors,
      badge: 'Passage Boundaries',
    },
    {
      num: '03',
      title: 'Dual Query Routing: Specific vs. Broad Intent',
      desc: 'An automated router analyzes your question. Specific lookup queries execute precision Hybrid Search (BM25 + Dense Vector RRF fusion + Cross-Encoder reranking). Broad queries ("summarize this document") execute Map-Reduce summarization across all chunks and cache the result keyed by tenant and document SHA-256 hash.',
      icon: Search,
      badge: 'BM25 + Dense Vector RRF',
    },
    {
      num: '04',
      title: 'Local Privacy & PII Scrubbing',
      desc: 'Personal details (names, emails, phone numbers, IP addresses) are redacted locally on your machine before excerpts are processed by LLM APIs, ensuring zero personal data exposure to third-party endpoints.',
      icon: ShieldCheck,
      badge: 'Zero PII Leakage',
    },
    {
      num: '05',
      title: 'Grounded Answer with Page Citations',
      desc: 'The AI generates a clear answer strictly grounded in your text. Every claim includes a clickable inline citation pointing directly to the source page and excerpt in the split-screen inspector.',
      icon: CheckCircle2,
      badge: 'Verifiable Evidence',
    },
  ];

  const differentiators = [
    {
      title: 'No Silent Document Truncation',
      desc: 'Pasting long PDFs into standard chat boxes often quietly cuts off pages past token limits. Here, 100% of your document is indexed and fully searchable.',
      icon: FileText,
    },
    {
      title: 'Clickable Source Page Citations',
      desc: 'Standard chatbots ask you to trust their answers on faith. Every claim generated here carries a clickable citation pointing directly to the exact page excerpt.',
      icon: Bookmark,
    },
    {
      title: 'Multi-Tenant Isolation',
      desc: 'Your documents live in an isolated browser-session container. No unauthenticated third party can query or view your indexed files.',
      icon: Lock,
    },
    {
      title: 'BYOK & Offline Support',
      desc: 'Use default free Groq 70B models, connect your own API keys (OpenAI, Anthropic, Gemini), or run completely offline using local Ollama instances.',
      icon: Cpu,
    },
  ];

  return (
    <div className="w-full h-full overflow-x-hidden overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans antialiased select-none flex flex-col justify-between">
      <main className="w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-12 sm:space-y-16">
        {/* Contextual Header */}
        <div className="space-y-4 border-b border-warmborder pb-6 sm:pb-8">
          <div className="flex items-center justify-between">
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 text-xs font-mono font-bold text-terracotta-700 hover:text-terracotta-800 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Home</span>
            </Link>
            <span className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
              <BookOpen className="w-4 h-4 text-terracotta-600" />
              Operational Guide
            </span>
          </div>

          <div className="space-y-2">
            <h1 className="font-serif font-bold text-3xl sm:text-4xl lg:text-5xl text-charcoal-900 tracking-tight">
              How It Works
            </h1>
            <p className="font-serif italic text-charcoal-700 text-base sm:text-lg max-w-2xl">
              From document ingestion to cited, verifiable answers — step by step.
            </p>
          </div>
        </div>

        {/* Step-by-Step Explanation Section */}
        <div className="space-y-6">
          <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-terracotta-600" /> The 5-Step RAG Pipeline
          </div>

          <div className="space-y-4">
            {steps.map((step, idx) => {
              const Icon = step.icon;
              return (
                <div
                  key={idx}
                  className="p-5 sm:p-6 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs flex flex-col sm:flex-row items-start gap-4 sm:gap-5 transition-all hover:border-terracotta-600/40 hover:shadow-xs group"
                >
                  <div className="flex sm:flex-col items-center justify-between sm:justify-start w-full sm:w-auto shrink-0 border-b sm:border-b-0 border-warmborder/60 pb-3 sm:pb-0">
                    <span className="font-mono font-bold text-xs text-terracotta-700 bg-terracotta-100 px-2 py-0.5 rounded border border-terracotta-600/20">
                      Step {step.num}
                    </span>
                    <div className="w-10 h-10 rounded-xl bg-parchment-100 border border-warmborder flex items-center justify-center sm:mt-2 text-terracotta-700 group-hover:scale-105 transition-transform">
                      <Icon className="w-5 h-5" />
                    </div>
                  </div>

                  <div className="space-y-2 flex-1 min-w-0">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h2 className="font-serif font-bold text-base sm:text-xl text-charcoal-900">
                        {step.title}
                      </h2>
                      <span className="font-mono text-[10px] text-terracotta-700 bg-terracotta-100/80 px-2 py-0.5 rounded font-semibold border border-terracotta-600/20">
                        {step.badge}
                      </span>
                    </div>
                    <p className="text-xs sm:text-sm text-charcoal-700 leading-relaxed font-sans">
                      {step.desc}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Technical Deep-Dive Box */}
        <div className="p-6 sm:p-8 rounded-3xl bg-parchment-200/60 border border-warmborder space-y-4 shadow-2xs">
          <div className="flex items-center gap-2 text-xs uppercase font-mono font-bold tracking-wider text-terracotta-700">
            <Cpu className="w-4 h-4 text-terracotta-700" /> Technical Underpinnings
          </div>
          <h3 className="font-serif font-bold text-xl sm:text-2xl text-charcoal-900">
            Reciprocal Rank Fusion (RRF) & Neural Cross-Encoding
          </h3>
          <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
            When you submit a question, lexical BM25 matching retrieves exact keyword hits, while dense embeddings retrieve semantic vector neighbors. Results are combined using <strong>Reciprocal Rank Fusion (RRF)</strong> score formula:
          </p>
          <div className="p-3.5 rounded-xl bg-parchment-50 border border-warmborder font-mono text-xs text-charcoal-900 overflow-x-auto">
            RRF_Score(d) = Σ [ 1 / (60 + rank_bm25(d)) ] + Σ [ 1 / (60 + rank_vector(d)) ]
          </div>
          <p className="font-sans text-xs text-charcoal-700 leading-relaxed">
            Candidate passages are then re-scored by a neural Cross-Encoder model to select only top relevant excerpts before feeding context into the LLM synthesis window.
          </p>
        </div>

        {/* Differentiators Section */}
        <div className="space-y-6">
          <div className="space-y-2">
            <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-terracotta-600" /> Why Ask My Documents?
            </div>
            <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight">
              Built specifically for evidence-based document work.
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 text-sm font-sans">
            {differentiators.map((item, i) => {
              const Icon = item.icon;
              return (
                <div key={i} className="p-5 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs space-y-2 hover:border-terracotta-600/40 transition-all">
                  <h3 className="font-serif font-bold text-base text-charcoal-900 flex items-center gap-2">
                    <Icon className="w-4 h-4 text-terracotta-600 shrink-0" />
                    {item.title}
                  </h3>
                  <p className="text-xs text-charcoal-700 leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Bottom CTA Banner */}
        <div className="pt-6 border-t border-warmborder text-center space-y-4">
          <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900">
            Ready to try it?
          </h2>
          <p className="font-sans text-sm text-charcoal-700">
            Upload your documents and get grounded answers with source citations in seconds.
          </p>
          <div className="pt-2">
            <Link
              to="/workspace"
              className="inline-flex items-center justify-center gap-2 min-h-[48px] px-8 py-3 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 active:bg-terracotta-800 text-parchment-50 font-serif font-bold text-base shadow-sm hover:shadow transition-all active:scale-[0.98]"
            >
              <span>Launch Workspace</span>
              <ArrowRight className="w-4 h-4 shrink-0" />
            </Link>
          </div>
        </div>
      </main>

      <LandingFooter />
    </div>
  );
}
