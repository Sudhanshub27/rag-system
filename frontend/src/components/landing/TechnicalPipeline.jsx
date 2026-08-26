import React from 'react';
import { Link } from 'react-router-dom';
import {
  FileText,
  Scissors,
  Search,
  SlidersHorizontal,
  CheckCircle2,
  Bookmark,
  ArrowRight,
  Sparkles,
} from 'lucide-react';

export default function TechnicalPipeline() {
  const pipelineSteps = [
    {
      num: '01',
      title: 'Documents',
      subtitle: 'Ingestion',
      desc: 'PDF, Word, Excel, CSV, JSON, MD, TXT',
      icon: FileText,
    },
    {
      num: '02',
      title: 'Parse & Chunk',
      subtitle: 'Passage Boundaries',
      desc: '250–512 token chunks with sentence preservation',
      icon: Scissors,
    },
    {
      num: '03',
      title: 'Hybrid Search',
      subtitle: 'BM25 + Dense RRF',
      desc: 'Lexical BM25 fused with ChromaDB vectors',
      icon: Search,
    },
    {
      num: '04',
      title: 'Reranking',
      subtitle: 'Cross-Encoder',
      desc: 'Neural scoring for exact query relevance',
      icon: SlidersHorizontal,
    },
    {
      num: '05',
      title: 'Grounded Answer',
      subtitle: 'Context Synthesis',
      desc: 'Strictly anchored in retrieved passages',
      icon: CheckCircle2,
    },
    {
      num: '06',
      title: 'Source Citation',
      subtitle: 'Inspector View',
      desc: 'Inline badges linking to verbatim excerpts',
      icon: Bookmark,
    },
  ];

  return (
    <section className="w-full bg-parchment-50 border-b border-[#C8BCA8]/70 py-8 sm:py-10 md:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
          <div className="space-y-1.5 max-w-2xl">
            <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-terracotta-600" />
              Retrieval Architecture
            </div>
            <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight">
              Built for retrieval, not just conversation.
            </h2>
            <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
              Standard chat windows quietly truncate long files. Ask My Documents indexes full documents through a 6-stage RAG pipeline.
            </p>
          </div>

          <Link
            to="/how-it-works"
            className="inline-flex items-center gap-1 font-sans font-semibold text-xs text-terracotta-700 hover:text-terracotta-800 transition-colors group shrink-0"
          >
            <span>See full process</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>

        {/* Compact 6-Stage Stepper Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
          {pipelineSteps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div
                key={idx}
                className="p-3.5 rounded-xl bg-parchment-100/90 border border-warmborder shadow-2xs hover:border-terracotta-600/40 hover:shadow-xs transition-all space-y-2 group flex flex-col justify-between"
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-[10px] text-terracotta-700 bg-terracotta-100/90 px-1.5 py-0.2 rounded border border-terracotta-600/20">
                      {step.num}
                    </span>
                    <div className="w-6 h-6 rounded-md bg-parchment-50 border border-warmborder flex items-center justify-center text-terracotta-700 group-hover:scale-105 transition-transform">
                      <Icon className="w-3 h-3" />
                    </div>
                  </div>

                  <div>
                    <h3 className="font-serif font-bold text-xs sm:text-sm text-charcoal-900 leading-snug">
                      {step.title}
                    </h3>
                    <div className="font-mono text-[9px] text-terracotta-700 font-semibold mt-0.5">
                      {step.subtitle}
                    </div>
                  </div>

                  <p className="font-sans text-[10px] text-charcoal-600 leading-normal">
                    {step.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
