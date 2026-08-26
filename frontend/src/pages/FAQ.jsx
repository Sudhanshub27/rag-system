import React from 'react';
import { Link } from 'react-router-dom';
import {
  HelpCircle,
  FileText,
  Cookie,
  HardDrive,
  Smartphone,
  CheckCircle,
  ShieldCheck,
  FileSpreadsheet,
  Code,
  FileCode,
  FileBox,
  ArrowRight,
  ArrowLeft,
} from 'lucide-react';
import LandingFooter from '../components/landing/LandingFooter';

export default function FAQ() {
  const fileCategories = [
    {
      title: 'Documents',
      icon: FileText,
      formats: [
        { name: 'PDF Document', ext: '.pdf' },
        { name: 'Microsoft Word', ext: '.docx' },
        { name: 'Legacy Word', ext: '.doc' },
      ],
    },
    {
      title: 'Spreadsheets & Tabular',
      icon: FileSpreadsheet,
      formats: [
        { name: 'Microsoft Excel', ext: '.xlsx' },
        { name: 'Legacy Excel', ext: '.xls' },
        { name: 'CSV File', ext: '.csv' },
        { name: 'TSV File', ext: '.tsv' },
      ],
    },
    {
      title: 'Structured & Data',
      icon: Code,
      formats: [
        { name: 'JSON Object', ext: '.json' },
        { name: 'JSON Lines', ext: '.jsonl' },
        { name: 'XML Data', ext: '.xml' },
        { name: 'YAML Config', ext: '.yaml, .yml' },
      ],
    },
    {
      title: 'Text, Web & Code',
      icon: FileCode,
      formats: [
        { name: 'Plain Text', ext: '.txt' },
        { name: 'Markdown', ext: '.md, .markdown' },
        { name: 'reStructuredText', ext: '.rst' },
        { name: 'Web HTML', ext: '.html, .htm' },
        { name: 'Log Files', ext: '.log' },
      ],
    },
  ];

  const faqs = [
    {
      q: 'Which LLM providers are supported, and is my data ever used for training?',
      a: 'We strictly support LLM options with documented zero-training policies: Groq API (Free & Paid zero-training policy), Ollama (100% Offline Local), OpenAI API (never trained on API data), Anthropic Commercial API (never trained on API data), and DeepSeek API (zero model training policy). Consumer free tiers that log data for model training are excluded.',
      icon: ShieldCheck,
    },
    {
      q: 'Can someone steal or access my documents if they see my Tenant ID?',
      a: 'No — access to your indexed documents and queries is strictly enforced by server-side session cookies. Knowing or copying a Tenant ID string does not grant third parties permission to query or view your documents.',
      icon: ShieldCheck,
    },
    {
      q: 'Will it make things up (hallucinate)?',
      a: 'The system is designed to answer strictly from passages retrieved from your uploaded documents. If relevant information is not found in your files, it indicates that rather than generating an ungrounded guess.',
      icon: CheckCircle,
    },
    {
      q: 'What file types are supported?',
      isStructuredFormats: true,
      icon: FileBox,
    },
    {
      q: 'What happens if I clear my cookies?',
      a: "You'll lose access to the documents tied to that browser session — there's no account to recover them from, since no account was ever created. This is explained in more detail on the Privacy page.",
      icon: Cookie,
    },
    {
      q: 'Is there a limit on document size or number of documents?',
      a: 'Supported up to 100MB per file. There is no strict limit on the number of documents you can upload into your private knowledge base.',
      icon: HardDrive,
    },
    {
      q: 'Can I use this on my phone and see the same documents I uploaded on my laptop?',
      a: "Not currently — your documents are tied to the browser session key created on that device, not to a user login account, so a different browser or device won't share the same local session key.",
      icon: Smartphone,
    },
  ];

  return (
    <div className="w-full h-full overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans antialiased select-none">
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-10 sm:space-y-14">
        {/* Contextual Header */}
        <div className="space-y-4 border-b border-warmborder pb-6 sm:pb-8">
          <div className="flex items-center justify-between">
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 text-xs font-mono font-bold text-terracotta-700 hover:text-terracotta-800 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Home</span>
            </Link>
            <span className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
              <HelpCircle className="w-4 h-4 text-terracotta-600" />
              Frequently Asked Questions
            </span>
          </div>

          <div className="space-y-2">
            <h1 className="font-serif font-bold text-3xl sm:text-4xl text-charcoal-900 tracking-tight">
              FAQ
            </h1>
            <p className="font-serif italic text-charcoal-700 text-base sm:text-lg">
              Common questions about document isolation, Tenant IDs, file formats, and privacy.
            </p>
          </div>
        </div>

        {/* FAQ Cards */}
        <div className="space-y-4 font-sans text-sm">
          {faqs.map((item, index) => {
            const Icon = item.icon;
            return (
              <div
                key={index}
                className="p-5 sm:p-6 rounded-2xl bg-parchment-50 border border-warmborder shadow-2xs space-y-3 hover:border-terracotta-600/40 transition-all"
              >
                <h2 className="font-serif font-bold text-base sm:text-lg text-charcoal-900 flex items-start gap-3">
                  <div className="w-7 h-7 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0 mt-0.5">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="pt-0.5">{item.q}</span>
                </h2>

                {item.isStructuredFormats ? (
                  <div className="pl-0 sm:pl-10 space-y-4 pt-1">
                    <p className="text-xs sm:text-sm text-charcoal-700 font-sans leading-relaxed">
                      We support <strong>19 distinct file formats</strong> across documents, spreadsheets, structured data, and code markup. All formats are parsed, chunked, and indexed with equal retrieval accuracy:
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {fileCategories.map((cat, cIdx) => {
                        const CatIcon = cat.icon;
                        return (
                          <div
                            key={cIdx}
                            className="p-4 rounded-xl bg-parchment-100/70 border border-warmborder/80 space-y-2"
                          >
                            <div className="flex items-center gap-2 font-serif font-bold text-xs text-charcoal-900 border-b border-warmborder/60 pb-2">
                              <CatIcon className="w-4 h-4 text-terracotta-600 shrink-0" />
                              <span>{cat.title}</span>
                            </div>
                            <div className="space-y-1.5 pt-1">
                              {cat.formats.map((fmt, fIdx) => (
                                <div
                                  key={fIdx}
                                  className="flex items-center justify-between text-xs text-charcoal-800"
                                >
                                  <span>{fmt.name}</span>
                                  <code className="px-1.5 py-0.5 rounded bg-parchment-200 text-charcoal-900 font-mono text-[11px] font-semibold border border-warmborder">
                                    {fmt.ext}
                                  </code>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <p className="text-charcoal-700 leading-relaxed font-sans pl-0 sm:pl-10 text-xs sm:text-sm">
                    {item.a}
                  </p>
                )}
              </div>
            );
          })}
        </div>

        {/* Bottom CTA Block */}
        <div className="border-t border-warmborder pt-10 pb-4">
          <div className="p-8 sm:p-10 rounded-3xl bg-parchment-200/60 border border-warmborder text-center space-y-4 shadow-xs">
            <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900">
              Try Ask My Documents
            </h2>
            <p className="font-sans text-xs sm:text-sm text-charcoal-700 max-w-md mx-auto">
              Start querying your files instantly with zero account setup.
            </p>
            <div className="pt-2">
              <Link
                to="/workspace"
                className="inline-flex items-center justify-center gap-2 min-h-[44px] px-6 py-3 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 active:bg-terracotta-800 text-parchment-50 font-serif font-bold text-sm sm:text-base shadow-sm hover:shadow transition-all text-center"
              >
                <span>Try Ask My Documents</span>
                <ArrowRight className="w-4 h-4 shrink-0" />
              </Link>
            </div>
          </div>
        </div>
      </main>

      <LandingFooter />
    </div>
  );
}
