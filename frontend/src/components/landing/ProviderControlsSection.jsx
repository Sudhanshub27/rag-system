import React from 'react';
import { Sliders, Cpu, Key, Lock } from 'lucide-react';

export default function ProviderControlsSection() {
  const capabilities = [
    {
      title: 'Provider Choice & BYOK',
      desc: 'Use default zero-training Groq API, connect your own keys (OpenAI, Anthropic, DeepSeek, Gemini), or run 100% offline via local Ollama.',
      icon: Cpu,
      badge: 'Multi-LLM / Local',
    },
    {
      title: 'Advanced Retrieval Controls',
      desc: 'Toggle HyDE (Hypothetical Document Embeddings) and Multi-Query expansion, or trigger Map-Reduce document summarization.',
      icon: Sliders,
      badge: 'HyDE & Multi-Query',
    },
    {
      title: 'Local PII Anonymization',
      desc: 'Parse and mask personal identifiers (names, emails, phone numbers, IP addresses) locally on your device before sending API payloads.',
      icon: Lock,
      badge: 'Local Redaction',
    },
  ];

  return (
    <section className="w-full bg-parchment-100 border-b border-[#C8BCA8]/70 py-8 sm:py-10 md:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="max-w-2xl space-y-1.5">
          <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
            <Key className="w-3.5 h-3.5 text-terracotta-600" />
            Configurable Controls
          </div>
          <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight">
            Flexible providers & retrieval controls
          </h2>
          <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
            Tailor the RAG engine to your exact workflow. Choose your LLM provider, fine-tune query expansion strategies, or anonymize sensitive PII locally.
          </p>
        </div>

        {/* 3 Control Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {capabilities.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div
                key={idx}
                className="p-5 rounded-xl bg-parchment-50 border border-warmborder shadow-2xs space-y-3 hover:border-terracotta-600/40 transition-all flex flex-col justify-between"
              >
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between">
                    <div className="w-9 h-9 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="font-mono text-[10px] font-bold text-terracotta-700 bg-terracotta-100 px-2 py-0.5 rounded border border-terracotta-600/20">
                      {item.badge}
                    </span>
                  </div>

                  <h3 className="font-serif font-bold text-base text-charcoal-900">
                    {item.title}
                  </h3>

                  <p className="font-sans text-xs text-charcoal-700 leading-relaxed">
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
