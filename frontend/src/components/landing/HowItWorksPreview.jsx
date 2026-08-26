import React from 'react';
import { Link } from 'react-router-dom';
import { FileUp, MessageSquare, SearchCheck, ArrowRight } from 'lucide-react';

export default function HowItWorksPreview() {
  const steps = [
    {
      step: '01',
      title: 'Upload',
      desc: 'Add PDFs, Word files, spreadsheets, JSON, Markdown, text and more.',
      icon: FileUp,
    },
    {
      step: '02',
      title: 'Ask',
      desc: 'Ask questions naturally across the documents you’ve uploaded.',
      icon: MessageSquare,
    },
    {
      step: '03',
      title: 'Verify',
      desc: 'Inspect the source passages behind the answer.',
      icon: SearchCheck,
    },
  ];

  return (
    <section className="py-12 sm:py-16 md:py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-warmborder">
      <div className="space-y-10">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div className="space-y-2">
            <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600">
              Operational Workflow
            </div>
            <h2 className="font-serif font-bold text-2xl sm:text-3xl md:text-4xl text-charcoal-900 tracking-tight">
              Three steps to verifiable answers
            </h2>
          </div>

          <Link
            to="/how-it-works"
            className="inline-flex items-center gap-1.5 font-sans font-semibold text-xs sm:text-sm text-terracotta-700 hover:text-terracotta-800 transition-colors group shrink-0"
          >
            <span>See the full process</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>

        {/* 3 Step Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {steps.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div
                key={idx}
                className="p-6 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs relative flex flex-col justify-between space-y-4 hover:border-terracotta-600/40 transition-all"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-xl text-terracotta-600">
                      {item.step}
                    </span>
                    <div className="w-9 h-9 rounded-lg bg-terracotta-100/60 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700">
                      <Icon className="w-4 h-4" />
                    </div>
                  </div>

                  <h3 className="font-serif font-bold text-lg text-charcoal-900">
                    {item.title}
                  </h3>

                  <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
                    {item.desc}
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
