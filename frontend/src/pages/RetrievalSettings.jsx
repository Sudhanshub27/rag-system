import React from 'react';
import { Sliders, Layers, Search, Cpu, CheckSquare, ArrowLeft, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import LandingFooter from '../components/landing/LandingFooter';

export default function RetrievalSettings() {
  return (
    <div className="w-full h-full overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans select-none flex flex-col justify-between">
      <main className="w-full max-w-4xl mx-auto px-4 sm:px-8 py-6 sm:py-10 space-y-6 sm:space-y-10 flex-1">
        {/* Contextual Header */}
        <div className="border-b border-warmborder pb-5 sm:pb-6 space-y-3">
          <div className="flex items-center justify-between">
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 text-xs font-mono font-bold text-terracotta-700 hover:text-terracotta-800 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Home</span>
            </Link>
            <div className="flex items-center gap-1.5 text-terracotta-700 font-bold text-xs uppercase tracking-wider font-mono">
              <Sliders className="w-3.5 h-3.5 shrink-0" /> Features & Controls
            </div>
          </div>

          <div className="space-y-1">
            <h1 className="font-serif font-bold text-2xl sm:text-4xl text-charcoal-900 tracking-tight">
              Retrieval Features & Controls
            </h1>
            <p className="font-serif italic text-charcoal-800 text-sm sm:text-base">
              Configurable search mechanisms, multi-query expansion, HyDE, and local PII anonymization.
            </p>
          </div>
        </div>

        <p className="text-sm text-charcoal-800 leading-relaxed font-sans">
          The default configuration works out-of-the-box for most questions. Here is how each feature works, why it helps, and how dual query intent is routed automatically.
        </p>

        {/* Feature 0: Dual Query Routing */}
        <div className="space-y-3 p-6 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs">
          <h2 className="font-serif font-bold text-lg sm:text-xl text-charcoal-900 flex items-center gap-2">
            <Sliders className="w-5 h-5 text-terracotta-600 shrink-0" />
            Automatic Dual Query Routing (Narrow vs. Broad)
          </h2>
          <div className="space-y-2 text-xs sm:text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>How it works:</strong> When you enter a question, the system's <code>QueryRouter</code> automatically classifies your intent using keyword pattern matching and BM25 score-shape heuristics:</p>
            <ul className="list-disc pl-5 space-y-1.5 text-xs text-charcoal-700">
              <li><strong>Narrow Intent (Specific Lookups):</strong> Questions asking for exact facts, dates, codes, or specific sections route through Hybrid Vector + BM25 search and Reranking.</li>
              <li><strong>Broad Intent (Document Summarization):</strong> Queries like <em>"summarize this document"</em> or <em>"explain the whole PDF"</em> route to a specialized Map-Reduce summarizer. The resulting summary is cached on disk keyed by your Tenant ID and document content SHA-256 hash.</li>
            </ul>
          </div>
        </div>

        {/* Feature 1: Multi-Query */}
        <div className="space-y-3 p-6 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs">
          <h2 className="font-serif font-bold text-lg sm:text-xl text-charcoal-900 flex items-center gap-2">
            <Layers className="w-5 h-5 text-terracotta-600 shrink-0" />
            1. Multi-Query Expansion
          </h2>
          <div className="space-y-2 text-xs sm:text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>What it is:</strong> Instead of searching with only your exact wording, the system rephrases your question two or three different ways in the background, then searches for all of them.</p>
            <p><strong>Why it helps:</strong> If your question is vague, oddly worded, or uses different terms than the document does, a single search can miss the relevant passage entirely. Multiple phrasings widen the net.</p>
            <p><strong>When to use it:</strong> Turn it on for broad or loosely-worded questions ("what's this about," "explain the process"). Leave it off for precise lookups where you already know the exact term used in the document.</p>
            <div className="p-3 bg-terracotta-100/50 border-l-2 border-terracotta-600 rounded-lg text-xs font-serif italic text-charcoal-900">
              <strong>What changes in the answer:</strong> You'll typically get more complete coverage of a topic, since more relevant passages get pulled in.
            </div>
          </div>
        </div>

        {/* Feature 2: HyDE */}
        <div className="space-y-3 p-6 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs">
          <h2 className="font-serif font-bold text-lg sm:text-xl text-charcoal-900 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-terracotta-600 shrink-0" />
            2. HyDE (Hypothetical Document Embeddings)
          </h2>
          <div className="space-y-2 text-xs sm:text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>What it is:</strong> Before searching, the system asks the model to write a short hypothetical answer to your question, then searches for real chunks that <em>resemble</em> that hypothetical answer — rather than searching directly on your question's wording.</p>
            <p><strong>Why it helps:</strong> Questions and answers are often phrased very differently. Searching by what an answer would <em>look like</em> often finds better matches than searching by the question itself.</p>
            <div className="p-3 bg-terracotta-100/50 border-l-2 border-terracotta-600 rounded-lg text-xs font-serif italic text-charcoal-900">
              <strong>What changes in the answer:</strong> Improves relevance on conceptual or "why/how" questions.
            </div>
          </div>
        </div>

        {/* Feature 3: Hybrid Search */}
        <div className="space-y-3 p-6 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs">
          <h2 className="font-serif font-bold text-lg sm:text-xl text-charcoal-900 flex items-center gap-2">
            <Search className="w-5 h-5 text-terracotta-600 shrink-0" />
            3. Hybrid Search (Keyword BM25 + Vector Search)
          </h2>
          <div className="space-y-2 text-xs sm:text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>What it is:</strong> Combines two search methods — BM25 keyword matching (finds exact terms, numbers, names) and semantic vector search (finds passages that mean the same thing even with different words) — then merges the results using Reciprocal Rank Fusion.</p>
            <div className="p-3 bg-terracotta-100/50 border-l-2 border-terracotta-600 rounded-lg text-xs font-serif italic text-charcoal-900">
              <strong>What changes in the answer:</strong> Reliable retrieval overall, especially for questions mixing specific numbers or codes with general concepts.
            </div>
          </div>
        </div>

        {/* Feature 4: Reranking */}
        <div className="space-y-3 p-6 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs">
          <h2 className="font-serif font-bold text-lg sm:text-xl text-charcoal-900 flex items-center gap-2">
            <CheckSquare className="w-5 h-5 text-terracotta-600 shrink-0" />
            4. Cross-Encoder Reranking
          </h2>
          <div className="space-y-2 text-xs sm:text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>What it is:</strong> After initial search returns candidate passages, a neural cross-encoder model re-scores them for relevance to your specific question and keeps only the best ones before generating an answer.</p>
            <div className="p-3 bg-terracotta-100/50 border-l-2 border-terracotta-600 rounded-lg text-xs font-serif italic text-charcoal-900">
              <strong>What changes in the answer:</strong> Answers tend to be more tightly grounded in the most relevant chunks with minimal noise.
            </div>
          </div>
        </div>

        {/* Bottom CTA Banner */}
        <div className="pt-6 border-t border-warmborder text-center space-y-3">
          <h2 className="font-serif font-bold text-xl sm:text-2xl text-charcoal-900">
            Ready to test these settings?
          </h2>
          <div>
            <Link
              to="/workspace"
              className="inline-flex items-center justify-center gap-2 min-h-[44px] px-6 py-2.5 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 active:bg-terracotta-800 text-parchment-50 font-serif font-bold text-sm shadow-sm hover:shadow transition-all text-center"
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
