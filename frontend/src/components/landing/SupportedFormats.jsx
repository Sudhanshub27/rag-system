import React from 'react';
import { FileText, FileSpreadsheet, Code, FileCode, Layers } from 'lucide-react';

export default function SupportedFormats() {
  const categories = [
    {
      title: 'Documents',
      icon: FileText,
      formats: [
        { name: 'PDF Document', ext: '.pdf', color: 'bg-red-100 text-red-700 border-red-200' },
        { name: 'Microsoft Word', ext: '.docx', color: 'bg-blue-100 text-blue-700 border-blue-200' },
        { name: 'Legacy Word', ext: '.doc', color: 'bg-sky-100 text-sky-700 border-sky-200' },
      ],
    },
    {
      title: 'Spreadsheets & Tabular',
      icon: FileSpreadsheet,
      formats: [
        { name: 'Microsoft Excel', ext: '.xlsx', color: 'bg-emerald-100 text-emerald-700 border-emerald-200' },
        { name: 'Legacy Excel', ext: '.xls', color: 'bg-emerald-50 text-emerald-800 border-emerald-200' },
        { name: 'CSV Data', ext: '.csv', color: 'bg-teal-100 text-teal-700 border-teal-200' },
        { name: 'TSV Data', ext: '.tsv', color: 'bg-teal-50 text-teal-800 border-teal-200' },
      ],
    },
    {
      title: 'Structured Data',
      icon: Code,
      formats: [
        { name: 'JSON Object', ext: '.json', color: 'bg-amber-100 text-amber-800 border-amber-200' },
        { name: 'JSON Lines', ext: '.jsonl', color: 'bg-amber-50 text-amber-900 border-amber-200' },
        { name: 'XML Data', ext: '.xml', color: 'bg-orange-100 text-orange-800 border-orange-200' },
        { name: 'YAML Config', ext: '.yaml', color: 'bg-yellow-100 text-yellow-800 border-yellow-200' },
      ],
    },
    {
      title: 'Text, Web & Code',
      icon: FileCode,
      formats: [
        { name: 'Plain Text', ext: '.txt', color: 'bg-stone-200 text-stone-800 border-stone-300' },
        { name: 'Markdown', ext: '.md', color: 'bg-purple-100 text-purple-700 border-purple-200' },
        { name: 'Web HTML', ext: '.html', color: 'bg-indigo-100 text-indigo-700 border-indigo-200' },
        { name: 'System Logs', ext: '.log', color: 'bg-rose-100 text-rose-700 border-rose-200' },
      ],
    },
  ];

  return (
    <section className="w-full bg-parchment-50 border-b border-[#C8BCA8]/70 py-8 sm:py-10 md:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="max-w-2xl space-y-1.5">
          <div className="text-xs uppercase font-mono font-bold tracking-wider text-terracotta-600 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-terracotta-600" />
            File Support
          </div>
          <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight">
            19 file formats. One unified search.
          </h2>
          <p className="font-sans text-xs sm:text-sm text-charcoal-700 leading-relaxed">
            Upload PDFs, spreadsheets, structured configurations, or developer logs. All formats are automatically parsed and indexed.
          </p>
        </div>

        {/* Categories Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {categories.map((cat, idx) => {
            const Icon = cat.icon;
            return (
              <div
                key={idx}
                className="p-4 rounded-xl bg-parchment-100/90 border border-warmborder shadow-2xs space-y-2.5 hover:border-terracotta-600/40 transition-all"
              >
                <div className="flex items-center gap-2 font-serif font-bold text-sm text-charcoal-900 border-b border-warmborder/70 pb-2">
                  <div className="w-7 h-7 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span>{cat.title}</span>
                </div>

                <div className="space-y-1.5 font-sans text-xs">
                  {cat.formats.map((fmt, fIdx) => (
                    <div
                      key={fIdx}
                      className="flex items-center justify-between text-charcoal-800 text-[11px]"
                    >
                      <span className="font-medium">{fmt.name}</span>
                      <code className={`px-1.5 py-0.2 rounded font-mono text-[10px] font-bold border ${fmt.color}`}>
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
    </section>
  );
}
