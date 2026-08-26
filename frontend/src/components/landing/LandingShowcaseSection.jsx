import React from 'react';
import { Link } from 'react-router-dom';
import {
  FileText,
  Bookmark,
  Sliders,
  ShieldCheck,
  ArrowRight,
  Layers,
  Sparkles,
  CheckCircle2,
} from 'lucide-react';

export default function LandingShowcaseSection() {
  const formats = [
    { ext: '.pdf', label: 'PDF Docs', color: 'bg-red-100 text-red-700 border-red-200' },
    { ext: '.docx', label: 'MS Word', color: 'bg-blue-100 text-blue-700 border-blue-200' },
    { ext: '.xlsx', label: 'Excel', color: 'bg-emerald-100 text-emerald-700 border-emerald-200' },
    { ext: '.csv', label: 'CSV Data', color: 'bg-teal-100 text-teal-700 border-teal-200' },
    { ext: '.json', label: 'JSON Data', color: 'bg-amber-100 text-amber-800 border-amber-200' },
    { ext: '.md', label: 'Markdown', color: 'bg-purple-100 text-purple-700 border-purple-200' },
    { ext: '.txt', label: 'Plain Text', color: 'bg-stone-200 text-stone-800 border-stone-300' },
    { ext: '.html', label: 'Web HTML', color: 'bg-indigo-100 text-indigo-700 border-indigo-200' },
  ];

  return (
    <section className="w-full bg-parchment-50 border-b border-[#C8BCA8]/70 py-12 sm:py-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-12">
        {/* Top Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1: Supported Formats */}
          <div className="p-6 rounded-2xl bg-parchment-100 border border-warmborder shadow-2xs space-y-4 hover:border-terracotta-600/40 transition-all flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-xl bg-terracotta-100/80 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700">
                <FileText className="w-5 h-5" />
              </div>
              <h3 className="font-serif font-bold text-xl text-charcoal-900">
                19 Supported File Formats
              </h3>
              <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
                Upload PDFs, Word, Excel, CSV, JSON, Markdown, or developer log files. All formats are automatically parsed and indexed.
              </p>

              <div className="flex flex-wrap gap-1.5 pt-2">
                {formats.map((fmt, i) => (
                  <span
                    key={i}
                    className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold border ${fmt.color}`}
                  >
                    {fmt.ext}
                  </span>
                ))}
              </div>
            </div>

            <div className="pt-2 border-t border-warmborder/60">
              <Link
                to="/how-it-works"
                className="inline-flex items-center gap-1 font-sans font-semibold text-xs text-terracotta-700 hover:text-terracotta-800 transition-colors group"
              >
                <span>How documents are parsed</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>

          {/* Card 2: Grounded Citations */}
          <div className="p-6 rounded-2xl bg-parchment-100 border border-warmborder shadow-2xs space-y-4 hover:border-terracotta-600/40 transition-all flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-xl bg-terracotta-100/80 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700">
                <Bookmark className="w-5 h-5" />
              </div>
              <h3 className="font-serif font-bold text-xl text-charcoal-900">
                Clickable Source Citations
              </h3>
              <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
                Never trust hallucinated answers. Every statement generated carries an inline citation badge pointing directly to exact page excerpts in the inspector.
              </p>

              <div className="p-3 rounded-xl bg-parchment-50 border border-warmborder font-sans text-xs text-charcoal-900 space-y-1.5">
                <div className="font-semibold text-[11px] text-terracotta-700 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Direct Page Anchor
                </div>
                <div className="text-[11px] leading-relaxed text-charcoal-700">
                  "...Revenue increased by 18% during Q3.{' '}
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded bg-terracotta-100 text-terracotta-700 font-mono text-[9px] font-bold border border-terracotta-600/30">
                    [Source 1 · p. 14]
                  </span>
                  "
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-warmborder/60">
              <Link
                to="/workspace"
                className="inline-flex items-center gap-1 font-sans font-semibold text-xs text-terracotta-700 hover:text-terracotta-800 transition-colors group"
              >
                <span>Try asking a question</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>

          {/* Card 3: Retrieval Controls & Privacy */}
          <div className="p-6 rounded-2xl bg-parchment-100 border border-warmborder shadow-2xs space-y-4 hover:border-terracotta-600/40 transition-all flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-xl bg-terracotta-100/80 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700">
                <Sliders className="w-5 h-5" />
              </div>
              <h3 className="font-serif font-bold text-xl text-charcoal-900">
                Retrieval Controls & Privacy
              </h3>
              <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
                Toggle HyDE, Multi-Query expansion, BM25 + Dense RRF vector search, and local PII anonymization in your private browser container.
              </p>

              <div className="space-y-1.5 font-mono text-[11px] text-charcoal-800">
                <div className="flex items-center justify-between p-1.5 rounded bg-parchment-50 border border-warmborder">
                  <span>Dual Query Router</span>
                  <span className="text-terracotta-700 font-bold">Active</span>
                </div>
                <div className="flex items-center justify-between p-1.5 rounded bg-parchment-50 border border-warmborder">
                  <span>Local PII Scrubbing</span>
                  <span className="text-terracotta-700 font-bold">Enabled</span>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-warmborder/60">
              <Link
                to="/retrieval-settings"
                className="inline-flex items-center gap-1 font-sans font-semibold text-xs text-terracotta-700 hover:text-terracotta-800 transition-colors group"
              >
                <span>Explore retrieval features</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
