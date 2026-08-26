import React from 'react';
import { Check, X, Layers, HelpCircle } from 'lucide-react';

export default function ComparisonSection() {
  const comparisons = [
    {
      feature: 'Document Parsing & Indexing',
      askMyDocs: 'Full document indexing (250-512 token chunks with sentence boundaries)',
      genericChat: 'Context/token limits (pages cut off when token budget is exceeded)',
    },
    {
      feature: 'Multi-Document Operations',
      askMyDocs: 'Multi-document retrieval & cross-file synthesis',
      genericChat: 'Manual file attachment per prompt; loses cross-file context',
    },
    {
      feature: 'Verifiable Source Grounding',
      askMyDocs: 'Exact page & excerpt citations with side-by-side inspector',
      genericChat: 'Less transparent source grounding (risk of hallucinated details)',
    },
    {
      feature: 'Workspace Persistence',
      askMyDocs: 'Persistent session workspace; return anytime to ask new queries',
      genericChat: 'Chat history dependent; context cleared on new session',
    },
    {
      feature: 'Retrieval Controls',
      askMyDocs: 'Configurable HyDE, Multi-Query expansion, and debug rerank scores',
      genericChat: 'Opaque black-box retrieval logic without user controls',
    },
  ];

  return (
    <section className="w-full bg-parchment-50 border-b border-[#C8BCA8]/70 py-8 sm:py-10 md:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Section Header */}
        <div className="max-w-2xl space-y-1.5">
          <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-terracotta-600" />
            Architectural Comparison
          </div>
          <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight">
            Why not just paste a PDF into a chatbot?
          </h2>
          <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
            General conversational chat interfaces are optimized for creative writing. Document intelligence requires specialized RAG indexing.
          </p>
        </div>

        {/* Comparison Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Ask My Documents Column */}
          <div className="p-5 rounded-xl bg-parchment-100 border-2 border-terracotta-600/30 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-2.5 border-b border-warmborder">
              <div className="flex items-center gap-2">
                <img src="/fav-icon.png" alt="Logo" className="w-5 h-5 object-contain" />
                <h3 className="font-serif font-bold text-base text-charcoal-900">
                  Ask My Documents
                </h3>
              </div>
              <span className="font-mono text-[9px] text-terracotta-700 bg-terracotta-100 px-2 py-0.5 rounded font-bold">
                Specialized RAG System
              </span>
            </div>

            <ul className="space-y-3 text-xs font-sans">
              {comparisons.map((c, i) => (
                <li key={i} className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center shrink-0 mt-0.5">
                    <Check className="w-3 h-3 stroke-[3]" />
                  </div>
                  <div>
                    <span className="font-semibold text-charcoal-900 block font-serif">
                      {c.feature}
                    </span>
                    <span className="text-charcoal-700 leading-relaxed text-[11px]">
                      {c.askMyDocs}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* Generic Chat Column */}
          <div className="p-5 rounded-xl bg-parchment-100/70 border border-warmborder space-y-4">
            <div className="flex items-center justify-between pb-2.5 border-b border-warmborder">
              <div className="flex items-center gap-2">
                <HelpCircle className="w-4 h-4 text-charcoal-500" />
                <h3 className="font-serif font-bold text-base text-charcoal-700">
                  Generic Chat Models
                </h3>
              </div>
              <span className="font-mono text-[9px] text-charcoal-500 bg-parchment-200 px-2 py-0.5 rounded">
                General LLM Windows
              </span>
            </div>

            <ul className="space-y-3 text-xs font-sans text-charcoal-600">
              {comparisons.map((c, i) => (
                <li key={i} className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-parchment-300/60 text-charcoal-500 flex items-center justify-center shrink-0 mt-0.5">
                    <X className="w-3 h-3 stroke-[2.5]" />
                  </div>
                  <div>
                    <span className="font-semibold text-charcoal-800 block font-serif">
                      {c.feature}
                    </span>
                    <span className="text-charcoal-600 leading-relaxed text-[11px]">
                      {c.genericChat}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
