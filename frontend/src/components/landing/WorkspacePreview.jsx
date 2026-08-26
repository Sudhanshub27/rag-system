import React from 'react';
import {
  FileText,
  Upload,
  ShieldCheck,
  Sliders,
  Search,
  Lock,
  Cpu,
  Trash2,
  Key,
} from 'lucide-react';

export default function WorkspacePreview() {
  return (
    <div className="w-full select-none">
      {/* Application Window Outer Frame */}
      <div className="rounded-2xl border border-[#C8BCA8] bg-parchment-50 shadow-xl overflow-hidden transition-all hover:border-terracotta-600/40">
        {/* Top Window Bar */}
        <div className="bg-[#DFD5C3] border-b border-[#C8BCA8] px-3.5 py-2 flex items-center justify-between font-mono text-[11px] text-charcoal-800">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-terracotta-600 inline-block" />
              <span className="w-2.5 h-2.5 rounded-full bg-amber-600 inline-block" />
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 inline-block" />
            </div>
            <span className="font-serif font-bold text-charcoal-900 text-xs sm:text-sm ml-2 flex items-center gap-1.5">
              <img src="/fav-icon.png" alt="Logo" className="w-4 h-4 object-contain" />
              Ask My Documents Workspace
            </span>
          </div>

          <div className="flex items-center gap-2 text-[10px]">
            <span className="hidden sm:inline font-sans text-charcoal-600">
              Session Active
            </span>
            <span className="text-terracotta-700 bg-parchment-50 px-2 py-0.5 rounded border border-warmborder font-bold flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
              RAG Active
            </span>
          </div>
        </div>

        {/* Workspace Body: 2-Column Responsive Layout matching real Sidebar & MainWorkspace */}
        <div className="grid grid-cols-1 sm:grid-cols-12 font-sans text-xs">
          {/* Left Column: Real Sidebar Mock */}
          <div className="sm:col-span-5 bg-parchment-200/90 border-b sm:border-b-0 sm:border-r border-warmborder p-2.5 space-y-2 flex flex-col justify-between">
            <div className="space-y-2">
              {/* Tenant Session Bar */}
              <div className="px-2 py-1 rounded-lg bg-parchment-50/90 border border-warmborder flex items-center justify-between text-[10px] font-mono text-charcoal-800">
                <span className="flex items-center gap-1.5 font-sans font-semibold text-charcoal-700 truncate">
                  <Lock className="w-3 h-3 text-terracotta-600 shrink-0" />
                  <span>Tenant:</span>
                  <span className="font-mono text-charcoal-900 font-bold truncate">session_a8f9...</span>
                </span>
                <span className="text-sage-700 font-bold bg-sage-600/15 px-1.5 py-0.2 rounded shrink-0">
                  Isolated
                </span>
              </div>

              {/* LLM Provider Selection */}
              <div className="p-2 rounded-xl bg-parchment-50/90 border border-warmborder space-y-0.5">
                <div className="text-[9px] font-bold text-charcoal-700 uppercase tracking-wider flex items-center gap-1">
                  <Key className="w-3 h-3 text-terracotta-700" /> LLM Provider
                </div>
                <div className="p-1 bg-white border border-warmborder rounded text-[10px] font-sans font-medium text-charcoal-900 flex justify-between items-center">
                  <span>Groq API (Free 70B & Zero-Training)</span>
                  <span className="text-[9px] text-sage-700 font-bold">✓ Active</span>
                </div>
              </div>

              {/* Drag & Drop File Upload */}
              <div className="p-2 rounded-xl border-2 border-dashed border-warmborder bg-parchment-50/80 hover:bg-parchment-50 flex flex-col items-center justify-center gap-0.5 text-center cursor-pointer">
                <div className="p-1 rounded-full bg-terracotta-600/10 text-terracotta-600">
                  <Upload className="w-3.5 h-3.5" />
                </div>
                <div className="text-[10px] text-charcoal-700">
                  <span className="font-bold text-terracotta-700">Click to upload</span> or drag files
                </div>
                <div className="text-[8px] text-charcoal-500 font-mono">PDF, Word, Excel, CSV, JSON, MD, TXT</div>
              </div>

              {/* Knowledge Index Statistics Banner */}
              <div className="p-1.5 rounded-lg bg-parchment-50/90 border border-warmborder flex items-center justify-between text-[10px]">
                <div className="flex items-center gap-1 font-bold text-charcoal-800">
                  <Cpu className="w-3.5 h-3.5 text-terracotta-600 shrink-0" />
                  <span>Knowledge Index</span>
                </div>
                <div className="flex items-center gap-1 font-mono text-[9px]">
                  <span className="px-1.5 py-0.2 rounded bg-parchment-200 text-charcoal-900 font-semibold border border-warmborder">
                    3 Docs
                  </span>
                  <span className="px-1.5 py-0.2 rounded bg-terracotta-600/15 text-terracotta-700 font-bold border border-terracotta-600/30">
                    42 Chunks
                  </span>
                </div>
              </div>

              {/* Real Indexed Documents List */}
              <div className="space-y-1">
                <div className="text-[9px] font-bold text-charcoal-700 uppercase tracking-wider flex justify-between items-center">
                  <span>Indexed Documents (3)</span>
                </div>

                <div className="p-1.5 rounded-lg bg-parchment-50 border border-warmborder flex items-center justify-between text-xs text-charcoal-900 font-sans shadow-2xs">
                  <span className="truncate flex items-center gap-1.5 pr-2" title="Annual Report 2025.pdf">
                    <FileText className="w-3.5 h-3.5 text-terracotta-700 shrink-0" />
                    <span className="truncate font-medium text-[10px]">Annual Report 2025.pdf</span>
                  </span>
                  <button className="text-charcoal-400 hover:text-rust-600 shrink-0" title="Delete document">
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>

                <div className="p-1.5 rounded-lg bg-parchment-50 border border-warmborder flex items-center justify-between text-xs text-charcoal-900 font-sans shadow-2xs">
                  <span className="truncate flex items-center gap-1.5 pr-2" title="Product Architecture.docx">
                    <FileText className="w-3.5 h-3.5 text-amber-700 shrink-0" />
                    <span className="truncate font-medium text-[10px]">Product Architecture.docx</span>
                  </span>
                  <button className="text-charcoal-400 hover:text-rust-600 shrink-0" title="Delete document">
                    <Trash2 className="w-3.5 h-3" />
                  </button>
                </div>

                <div className="p-1.5 rounded-lg bg-parchment-50 border border-warmborder flex items-center justify-between text-xs text-charcoal-900 font-sans shadow-2xs">
                  <span className="truncate flex items-center gap-1.5 pr-2" title="Research Notes.md">
                    <FileText className="w-3.5 h-3.5 text-emerald-700 shrink-0" />
                    <span className="truncate font-medium text-[10px]">Research Notes.md</span>
                  </span>
                  <button className="text-charcoal-400 hover:text-rust-600 shrink-0" title="Delete document">
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </div>

            {/* Sidebar Controls Footer */}
            <div className="pt-1.5 border-t border-warmborder/80 flex items-center justify-between text-[9px] text-charcoal-600 font-mono">
              <span className="flex items-center gap-1">
                <Sliders className="w-3 h-3 text-terracotta-600" /> Hybrid BM25 + RRF
              </span>
              <span className="text-sage-700 font-semibold">Active</span>
            </div>
          </div>

          {/* Right Column: Chat Workspace */}
          <div className="sm:col-span-7 bg-parchment-50 p-3 space-y-2.5 flex flex-col justify-between">
            <div className="space-y-2.5">
              {/* User Question */}
              <div className="flex justify-end">
                <div className="bg-charcoal-900 text-parchment-50 px-3 py-2 rounded-2xl rounded-tr-xs max-w-[92%] text-[10px] leading-relaxed shadow-xs font-medium">
                  What are the key findings regarding Q3 revenue and performance latency?
                </div>
              </div>

              {/* Assistant Answer Bubble: Summary + Bullet Points + Citations Bar */}
              <div className="bg-parchment-100/90 border border-warmborder p-3 rounded-2xl rounded-tl-xs space-y-2 text-[10px] leading-relaxed text-charcoal-900 shadow-2xs">
                <p className="text-charcoal-900 font-sans font-medium">
                  Based on your uploaded documents, here is the breakdown of Q3 performance and system benchmarks:
                </p>

                <ul className="space-y-1.5 list-disc pl-4 text-charcoal-800 font-sans leading-relaxed">
                  <li>
                    <strong className="text-charcoal-900">Revenue Growth:</strong> Enterprise subscription revenue expanded by 18% YoY driven by multi-year licensing contracts{' '}
                    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded bg-terracotta-100 text-terracotta-700 font-mono text-[9px] font-bold border border-terracotta-600/30 cursor-pointer shadow-2xs">
                      [Source 1 · p. 12]
                    </span>
                    .
                  </li>
                  <li>
                    <strong className="text-charcoal-900">Latency Optimization:</strong> Hybrid vector indexing reduced query processing latency by 35% under load{' '}
                    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded bg-terracotta-100 text-terracotta-700 font-mono text-[9px] font-bold border border-terracotta-600/30 cursor-pointer shadow-2xs">
                      [Source 2 · p. 27]
                    </span>
                    .
                  </li>
                  <li>
                    <strong className="text-charcoal-900">Retention:</strong> Customer retention reached 94.2% across active enterprise tiers{' '}
                    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded bg-terracotta-100 text-terracotta-700 font-mono text-[9px] font-bold border border-terracotta-600/30 cursor-pointer shadow-2xs">
                      [Source 3 · p. 31]
                    </span>
                    .
                  </li>
                </ul>

                {/* Real Sources Buttons Bar matching actual ChatPanel.jsx */}
                <div className="pt-2 border-t border-warmborder/70 flex flex-wrap items-center gap-1 text-[9px]">
                  <span className="text-charcoal-600 font-semibold font-sans">Sources (Click to inspect):</span>
                  <button className="px-1.5 py-0.2 rounded bg-terracotta-100/80 text-terracotta-700 font-mono text-[9px] font-bold border border-terracotta-600/30 flex items-center gap-1 hover:bg-terracotta-200 transition-colors cursor-pointer">
                    <FileText className="w-3 h-3 text-terracotta-700" />
                    Annual Report 2025.pdf (p.12)
                  </button>
                  <button className="px-1.5 py-0.2 rounded bg-terracotta-100/80 text-terracotta-700 font-mono text-[9px] font-bold border border-terracotta-600/30 flex items-center gap-1 hover:bg-terracotta-200 transition-colors cursor-pointer">
                    <FileText className="w-3 h-3 text-terracotta-700" />
                    Product Architecture.docx (p.27)
                  </button>
                </div>
              </div>
            </div>

            {/* Input Bar Mock */}
            <div className="pt-1">
              <div className="flex items-center gap-2 p-2 rounded-xl bg-parchment-100 border border-warmborder text-[10px] text-charcoal-500 shadow-2xs">
                <Search className="w-3.5 h-3.5 text-charcoal-500 shrink-0" />
                <span className="flex-1 font-sans">Ask a follow-up question about these documents...</span>
                <span className="px-1.5 py-0.2 rounded bg-parchment-200 text-charcoal-800 font-mono text-[9px] font-bold border border-warmborder">
                  ↵ Enter
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
