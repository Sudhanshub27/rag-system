import React from 'react';
import { Bookmark, CheckCircle, FileText, Sparkles, ExternalLink, ShieldCheck } from 'lucide-react';

export default function CitationShowcase() {
  return (
    <section className="w-full bg-parchment-50 border-b border-[#C8BCA8]/70 py-8 sm:py-10 md:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-parchment-200/50 border border-warmborder rounded-2xl p-5 sm:p-7 lg:p-8 shadow-xs">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-center">
            {/* Text Left */}
            <div className="lg:col-span-5 space-y-4">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-terracotta-100 text-terracotta-700 font-mono text-[10px] font-bold uppercase tracking-wider border border-terracotta-600/20">
                <Sparkles className="w-3 h-3 shrink-0" />
                Source Grounding
              </div>

              <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight leading-tight">
                Don’t just get an answer.{' '}
                <span className="text-terracotta-600 italic block">
                  See where it came from.
                </span>
              </h2>

              <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
                Standard AI chat models fill knowledge gaps with unverified web predictions. Ask My Documents strictly constrains answers to your uploaded text and provides inline citations for every assertion.
              </p>

              <div className="space-y-2 pt-1 text-xs font-sans text-charcoal-800">
                <div className="flex items-start gap-2">
                  <CheckCircle className="w-3.5 h-3.5 text-terracotta-600 shrink-0 mt-0.5" />
                  <span>Clickable citation badges linking directly to page numbers</span>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="w-3.5 h-3.5 text-terracotta-600 shrink-0 mt-0.5" />
                  <span>Side-by-side Source Inspector showing verbatim passage context</span>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="w-3.5 h-3.5 text-terracotta-600 shrink-0 mt-0.5" />
                  <span>Reranker & Relevance scores for full mathematical transparency</span>
                </div>
              </div>
            </div>

            {/* Citation Inspector Visual Card Right */}
            <div className="lg:col-span-7 space-y-3">
              {/* Generated Answer Box */}
              <div className="p-3.5 rounded-xl bg-parchment-50 border border-warmborder shadow-2xs space-y-2 text-xs">
                <div className="flex items-center justify-between border-b border-warmborder/60 pb-1.5">
                  <span className="font-serif font-bold text-[11px] text-charcoal-900 flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-terracotta-600" />
                    Generated Response
                  </span>
                  <span className="font-mono text-[9px] text-terracotta-700 bg-terracotta-100 px-1.5 py-0.2 rounded font-bold">
                    Grounded QA
                  </span>
                </div>

                <div className="font-sans text-xs text-charcoal-900 leading-relaxed">
                  "Revenue increased by 18% during the reporting period.{' '}
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded bg-terracotta-100 text-terracotta-700 font-mono text-[10px] font-bold border border-terracotta-600/40 shadow-2xs">
                    <Bookmark className="w-3 h-3 text-terracotta-600" />
                    [Source 1 · Annual Report.pdf, Page 14]
                  </span>
                  "
                </div>
              </div>

              {/* Visual Source Inspector Drawer Mock */}
              <div className="p-4 rounded-xl bg-parchment-100 border-2 border-terracotta-600/40 shadow-sm space-y-2.5 relative overflow-hidden">
                <div className="flex items-center justify-between border-b border-warmborder pb-2">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded bg-terracotta-600 text-parchment-50 flex items-center justify-center font-mono font-bold text-xs">
                      1
                    </div>
                    <div>
                      <div className="font-serif font-bold text-xs text-charcoal-900 flex items-center gap-1">
                        <FileText className="w-3 h-3 text-terracotta-600" />
                        Annual Report.pdf
                      </div>
                      <div className="font-mono text-[10px] text-charcoal-500">
                        Page 14 • Paragraph 3
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 font-mono text-[10px] text-terracotta-700 bg-parchment-50 px-2 py-0.5 rounded border border-warmborder font-bold">
                    Relevance: 0.96
                  </div>
                </div>

                <div className="space-y-1 font-sans text-xs text-charcoal-800 leading-relaxed bg-parchment-50 p-3 rounded-lg border border-warmborder">
                  <span className="text-charcoal-500 font-mono text-[9px] block uppercase">Verbatim Extracted Passage:</span>
                  <p className="text-[11px]">
                    "...Operational expansion was driven primarily by strong subscription renewal rates across enterprise client accounts.{' '}
                    <mark className="bg-terracotta-100 text-terracotta-900 font-semibold px-1 rounded border border-terracotta-600/30">
                      Revenue increased by 18% during the reporting period
                    </mark>
                    , outperforming baseline projections..."
                  </p>
                </div>

                <div className="flex items-center justify-between pt-0.5 font-mono text-[9px] text-charcoal-500">
                  <span>Vector Chunk ID: tenant_aa14_chunk_142</span>
                  <span className="text-terracotta-700 font-bold flex items-center gap-1">
                    Verified Source <ExternalLink className="w-3 h-3" />
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
