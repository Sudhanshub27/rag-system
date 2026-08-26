import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, ArrowLeft, Lock, FileCode, Check } from 'lucide-react';
import LandingFooter from '../components/landing/LandingFooter';

export default function License() {
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
              <Shield className="w-3.5 h-3.5 shrink-0" /> Code License
            </div>
          </div>

          <div className="space-y-1">
            <h1 className="font-serif font-bold text-2xl sm:text-4xl text-charcoal-900 tracking-tight">
              Source-Available License & Terms
            </h1>
            <p className="font-serif italic text-charcoal-800 text-sm sm:text-base">
              Copyright (c) 2026 Sudhanshu Batra. All Rights Reserved.
            </p>
          </div>
        </div>

        <div className="space-y-6 font-sans text-sm text-charcoal-900 leading-relaxed">
          <section className="p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs space-y-3">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <FileCode className="w-4 h-4 text-terracotta-600 shrink-0" />
              Open Code Inspection Notice
            </h2>
            <p className="text-xs sm:text-sm text-charcoal-700 leading-relaxed">
              The source code for Ask My Documents is publicly accessible on GitHub for transparency, security inspection, and technical review. Open code access does not constitute an open-source license under OSI definitions.
            </p>
          </section>

          <section className="p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs space-y-3">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <Lock className="w-4 h-4 text-terracotta-600 shrink-0" />
              Proprietary Restrictions
            </h2>
            <ul className="space-y-2 text-xs sm:text-sm text-charcoal-700">
              <li className="flex items-start gap-2">
                <Check className="w-4 h-4 text-terracotta-600 shrink-0 mt-0.5" />
                <span>Personal evaluation, local testing, and security auditing are permitted.</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="w-4 h-4 text-terracotta-600 shrink-0 mt-0.5" />
                <span>Commercial distribution, sublicensing, or public SaaS hosting without express written consent is prohibited.</span>
              </li>
            </ul>
          </section>
        </div>
      </main>

      <LandingFooter />
    </div>
  );
}
