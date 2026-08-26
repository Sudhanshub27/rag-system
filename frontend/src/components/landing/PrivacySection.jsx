import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheck, UserX, Database, Lock, ArrowRight } from 'lucide-react';

export default function PrivacySection() {
  const points = [
    {
      title: 'No account required',
      description: 'Your browser session identifies your workspace without asking for your name or email.',
      icon: UserX,
    },
    {
      title: 'Isolated document storage',
      description: 'Documents are kept in an isolated tenant/container for your session.',
      icon: Database,
    },
    {
      title: 'Local privacy protections',
      description: 'Personal identifiers can be processed locally before external LLM requests.',
      icon: Lock,
    },
  ];

  return (
    <section className="w-full bg-parchment-100 border-b border-[#C8BCA8]/70 py-8 sm:py-10 md:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
          <div className="space-y-1.5 max-w-2xl">
            <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-terracotta-600" />
              Privacy Architecture
            </div>
            <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight">
              Private by design.
            </h2>
          </div>

          <Link
            to="/privacy"
            className="inline-flex items-center gap-1 font-sans font-semibold text-xs text-terracotta-700 hover:text-terracotta-800 transition-colors group shrink-0"
          >
            <span>Read privacy architecture</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>

        {/* 3 Privacy Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {points.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div
                key={idx}
                className="p-5 rounded-xl bg-parchment-50 border border-warmborder shadow-2xs space-y-2.5 hover:border-terracotta-600/40 transition-all"
              >
                <div className="w-9 h-9 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                  <Icon className="w-4 h-4" />
                </div>
                <h3 className="font-serif font-bold text-base text-charcoal-900">
                  {item.title}
                </h3>
                <p className="font-sans text-xs text-charcoal-700 leading-relaxed">
                  {item.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
