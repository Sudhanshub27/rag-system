import React from 'react';
import { Files, Bookmark, Database, ShieldCheck } from 'lucide-react';

export default function ValuePropSection() {
  const features = [
    {
      title: 'Ask across documents',
      description: 'Search and reason across multiple uploaded documents simultaneously instead of opening files one by one.',
      icon: Files,
    },
    {
      title: 'Grounded answers',
      description: 'The system answers from retrieved document content and can indicate when relevant information isn’t found.',
      icon: Bookmark,
    },
    {
      title: 'Built for large documents',
      description: 'Documents are parsed, chunked, indexed, and retrieved instead of being blindly pasted into a chat box.',
      icon: Database,
    },
    {
      title: 'Privacy by architecture',
      description: 'Your documents are isolated to your browser session in dedicated tenant containers without requiring an account.',
      icon: ShieldCheck,
    },
  ];

  return (
    <section className="w-full bg-parchment-50 border-b border-[#C8BCA8]/70 py-12 sm:py-16 md:py-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-10 text-center sm:text-left">
        {/* Section Header */}
        <div className="max-w-3xl space-y-3">
          <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600">
            Core Capabilities
          </div>
          <h2 className="font-serif font-bold text-2xl sm:text-3xl md:text-4xl text-charcoal-900 tracking-tight">
            Your documents. Your questions. Evidence included.
          </h2>
          <p className="font-sans text-sm sm:text-base text-charcoal-700 leading-relaxed">
            Designed for technical documents, contracts, research papers, and complex reports where exact sourcing and precision are paramount.
          </p>
        </div>

        {/* 4 Feature Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {features.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div
                key={idx}
                className="p-5 sm:p-6 rounded-2xl bg-parchment-100/90 border border-warmborder shadow-2xs hover:border-terracotta-600/40 hover:shadow-xs transition-all flex flex-col justify-between space-y-4 group"
              >
                <div className="space-y-3">
                  <div className="w-10 h-10 rounded-xl bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 group-hover:scale-105 transition-transform shrink-0">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h3 className="font-serif font-bold text-base sm:text-lg text-charcoal-900">
                    {item.title}
                  </h3>
                  <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
                    {item.description}
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
