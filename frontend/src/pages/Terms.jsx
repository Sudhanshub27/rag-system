import React from 'react';
import { Link } from 'react-router-dom';
import { FileText, ArrowLeft, ShieldCheck, CheckCircle2, AlertCircle } from 'lucide-react';
import LandingFooter from '../components/landing/LandingFooter';

export default function Terms() {
  return (
    <div className="w-full h-full overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans antialiased select-none flex flex-col justify-between">
      <main className="w-full max-w-4xl mx-auto px-4 sm:px-8 py-6 sm:py-10 space-y-8 flex-1">
        {/* Header */}
        <div className="border-b border-warmborder pb-5 space-y-3">
          <div className="flex items-center justify-between">
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 text-xs font-mono font-bold text-terracotta-700 hover:text-terracotta-800 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Home</span>
            </Link>
            <div className="flex items-center gap-1.5 text-terracotta-700 font-bold text-xs uppercase tracking-wider font-mono">
              <FileText className="w-3.5 h-3.5 shrink-0" /> Terms & Conditions
            </div>
          </div>

          <div className="space-y-1">
            <h1 className="font-serif font-bold text-2xl sm:text-4xl text-charcoal-900 tracking-tight">
              Terms of Service & Usage Conditions
            </h1>
            <p className="font-serif italic text-charcoal-800 text-sm sm:text-base">
              Usage rules, system bounds, and browser-session operational terms.
            </p>
          </div>
        </div>

        <div className="space-y-6 font-sans text-sm text-charcoal-900 leading-relaxed">
          <section className="p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs space-y-3">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-terracotta-600 shrink-0" />
              1. Acceptance of Terms
            </h2>
            <p className="text-xs sm:text-sm text-charcoal-700 leading-relaxed">
              By accessing or using Ask My Documents, you agree to comply with these terms. This application is provided for private document analysis, search, and answer generation. No registration or account creation is required.
            </p>
          </section>

          <section className="p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs space-y-3">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-terracotta-600 shrink-0" />
              2. Acceptable Document Uploads
            </h2>
            <p className="text-xs sm:text-sm text-charcoal-700 leading-relaxed">
              You are responsible for ensuring that documents uploaded to your workspace session do not violate applicable copyright laws, contain malicious payloads, or expose unlawful materials. Files are parsed strictly for your browser session.
            </p>
          </section>

          <section className="p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs space-y-3">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-terracotta-600 shrink-0" />
              3. Service Limitations & Grounding Notice
            </h2>
            <p className="text-xs sm:text-sm text-charcoal-700 leading-relaxed">
              While Ask My Documents employs hybrid BM25 + Vector retrieval and Cross-Encoder reranking to anchor answers in your text, generated responses should be verified against cited source excerpts in the Inspector before making legal or financial decisions.
            </p>
          </section>
        </div>
      </main>

      <LandingFooter />
    </div>
  );
}
