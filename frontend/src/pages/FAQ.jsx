import React from 'react';
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
} from 'lucide-react';

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
      a: 'We strictly support only LLM options with documented zero-training policies: Groq API (Free & Paid zero-training policy), Ollama (100% Offline Local), OpenAI API (never trained on API data), Anthropic Commercial API (never trained on API data), and DeepSeek API (zero model training policy). Consumer free tiers that log data for model training (like Google AI Studio free tier) are intentionally excluded.',
      icon: ShieldCheck,
    },
    {
      q: 'Can someone steal or access my documents if they see my Tenant ID?',
      a: 'No — access to your indexed documents and queries is strictly enforced by server-side session cookies. Knowing or copying a Tenant ID string does not grant third parties permission to query or view your documents.',
      icon: ShieldCheck,
    },
    {
      q: 'Will it make things up (hallucinate)?',
      a: 'No — the system only answers from passages it actually retrieved from your uploaded documents. If nothing relevant is found, it tells you that, rather than guessing from general knowledge. Every claim in an answer is tied to a citation you can check yourself.',
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
      a: "Not currently — your documents are tied to the browser you uploaded them from via an anonymous browser session key, not to you as a logged-in user, so a different browser or device won't have access to the same data.",
      icon: Smartphone,
    },
  ];

  return (
    <div className="w-full h-full overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans px-8 py-10">
      <div className="max-w-3xl mx-auto space-y-10 pb-16">
        {/* Header */}
        <div className="border-b border-warmborder pb-6 space-y-2">
          <div className="flex items-center gap-2 text-terracotta-600 font-semibold text-xs uppercase tracking-wider font-sans">
            <HelpCircle className="w-4 h-4" /> Frequently Asked Questions
          </div>
          <h1 className="font-serif font-bold text-3xl text-charcoal-900 tracking-tight">
            FAQ
          </h1>
          <p className="font-serif italic text-charcoal-700 text-base">
            Common questions about document isolation, Tenant IDs, file formats, and privacy.
          </p>
        </div>

        {/* FAQ Cards */}
        <div className="space-y-4 font-sans text-sm">
          {faqs.map((item, index) => {
            const Icon = item.icon;
            return (
              <div
                key={index}
                className="p-6 rounded-xl bg-parchment-50 border border-warmborder shadow-xs space-y-3"
              >
                <h3 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2.5">
                  <Icon className="w-5 h-5 text-terracotta-600 shrink-0" />
                  <span>{item.q}</span>
                </h3>

                {item.isStructuredFormats ? (
                  <div className="pl-7 space-y-4 pt-1">
                    <p className="text-xs text-charcoal-700 font-sans leading-relaxed">
                      We support <strong>19 distinct file formats</strong> across documents, spreadsheets, structured data, and code markup. All formats are automatically parsed, chunked, and indexed with equal retrieval accuracy:
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {fileCategories.map((cat, cIdx) => {
                        const CatIcon = cat.icon;
                        return (
                          <div
                            key={cIdx}
                            className="p-3.5 rounded-lg bg-parchment-100/70 border border-warmborder/80 space-y-2"
                          >
                            <div className="flex items-center gap-2 font-serif font-bold text-xs text-charcoal-900 border-b border-warmborder/60 pb-1.5">
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
                  <p className="text-charcoal-700 leading-relaxed font-sans pl-7 flex-1">
                    {item.a}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
