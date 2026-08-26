import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, BookOpen, FileText, Bookmark, CheckCircle2 } from 'lucide-react';
import WorkspacePreview from './WorkspacePreview';

export default function HeroSection() {
  return (
    <section className="w-full min-h-[calc(100vh-3.5rem)] flex flex-col justify-center bg-parchment-100 border-b border-[#C8BCA8]/70 py-6 sm:py-8 md:py-10 px-4 sm:px-6 lg:px-8 overflow-hidden">
      <div className="max-w-7xl mx-auto w-full">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-center">
          {/* Left Column: Copy & Actions */}
          <div className="lg:col-span-5 space-y-4 text-left min-w-0">
            {/* Small Eyebrow */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-terracotta-100/80 border border-terracotta-600/20 text-terracotta-700 font-mono text-[11px] font-bold uppercase tracking-wider">
              <span className="w-2 h-2 rounded-full bg-terracotta-600 animate-pulse shrink-0" />
              <span>Private Document Intelligence</span>
            </div>

            {/* Main Heading */}
            <h1 className="font-serif font-bold text-3xl sm:text-4xl lg:text-5xl text-charcoal-900 tracking-tight leading-[1.15]">
              Ask your documents.{' '}
              <span className="text-terracotta-600 italic block sm:inline">
                Get answers you can verify.
              </span>
            </h1>

            {/* Supporting Text */}
            <p className="font-sans text-sm sm:text-base text-charcoal-700 leading-relaxed max-w-xl">
              Upload your documents, ask questions in plain language, and get grounded answers with citations that point back to the source.
            </p>

            {/* Action CTAs */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-1">
              <Link
                to="/workspace"
                className="inline-flex items-center justify-center gap-2 min-h-[46px] px-6 py-2.5 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 active:bg-terracotta-800 text-parchment-50 font-serif font-bold text-base shadow-sm hover:shadow transition-all active:scale-[0.98] text-center"
              >
                <span>Try Ask My Documents</span>
                <ArrowRight className="w-4 h-4 shrink-0" />
              </Link>

              <Link
                to="/how-it-works"
                className="inline-flex items-center justify-center gap-2 min-h-[46px] px-5 py-2.5 rounded-xl bg-parchment-50 hover:bg-parchment-200/60 border border-warmborder text-charcoal-800 hover:text-charcoal-900 font-sans font-semibold text-sm transition-colors text-center"
              >
                <BookOpen className="w-4 h-4 text-terracotta-700 shrink-0" />
                <span>How it works</span>
              </Link>
            </div>

            {/* Trust Line - Forced 1 Single Line */}
            <div className="pt-1 flex items-center gap-1.5 text-[10px] sm:text-[11px] font-mono text-charcoal-600 font-medium truncate">
              <span className="inline-flex items-center gap-1 shrink-0">
                <CheckCircle2 className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
                No signup
              </span>
              <span className="text-warmborder shrink-0">•</span>
              <span className="inline-flex items-center gap-1 shrink-0">
                <Bookmark className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
                Source citations
              </span>
              <span className="text-warmborder shrink-0">•</span>
              <span className="inline-flex items-center gap-1 shrink-0">
                <FileText className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
                Multi-doc search
              </span>
            </div>
          </div>

          {/* Right Column: Static Workspace Preview */}
          <div className="lg:col-span-7 w-full min-w-0">
            <WorkspacePreview />
          </div>
        </div>
      </div>
    </section>
  );
}
