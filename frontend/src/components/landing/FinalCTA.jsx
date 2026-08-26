import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, FileText } from 'lucide-react';

export default function FinalCTA() {
  return (
    <section className="w-full bg-parchment-100 border-b border-[#C8BCA8]/70 py-8 sm:py-10 md:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-parchment-200/60 border border-warmborder rounded-2xl p-6 sm:p-8 text-center space-y-4 shadow-xs relative overflow-hidden">
          {/* Subtle decorative background accent */}
          <div className="absolute -right-16 -top-16 w-48 h-48 bg-terracotta-100/40 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -left-16 -bottom-16 w-48 h-48 bg-terracotta-100/40 rounded-full blur-3xl pointer-events-none" />

          <div className="relative space-y-3 max-w-xl mx-auto">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-parchment-50 border border-warmborder text-terracotta-700 font-mono text-[10px] font-bold">
              <FileText className="w-3 h-3" />
              <span>Instant Document Workspace</span>
            </div>

            <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight leading-tight">
              Your documents are ready to answer.
            </h2>

            <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
              Upload a document and start asking questions.
            </p>

            <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
              <Link
                to="/workspace"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 min-h-[44px] px-6 py-2.5 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 active:bg-terracotta-800 text-parchment-50 font-serif font-bold text-sm shadow-sm hover:shadow transition-all active:scale-[0.98] text-center"
              >
                <span>Try it now</span>
                <ArrowRight className="w-4 h-4 shrink-0" />
              </Link>
            </div>

            <div className="pt-1 font-mono text-[10px] text-charcoal-500">
              No registration required • Instant browser session Q&A
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
