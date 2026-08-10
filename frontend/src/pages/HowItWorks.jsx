import React from 'react';
import { BookOpen, FileCheck, ShieldCheck, Database, Layers, CheckCircle2 } from 'lucide-react';

export default function HowItWorks() {
  return (
    <div className="w-full h-full overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans px-8 py-10">
      <div className="max-w-3xl mx-auto space-y-10 pb-16">
        {/* Header */}
        <div className="border-b border-warmborder pb-6 space-y-2">
          <div className="flex items-center gap-2 text-terracotta-600 font-semibold text-xs uppercase tracking-wider font-sans">
            <BookOpen className="w-4 h-4" /> System Architecture & Overview
          </div>
          <h1 className="font-serif font-bold text-3xl text-charcoal-900 tracking-tight">
            How It Works
          </h1>
          <p className="font-serif italic text-charcoal-700 text-base">
            Why grounded RAG retrieval beats pasting long PDFs directly into ChatGPT or Claude.
          </p>
        </div>

        {/* Core Differentiation Banner */}
        <div className="p-6 rounded-xl bg-parchment-50 border border-warmborder shadow-sm space-y-3">
          <h2 className="font-serif font-bold text-xl text-charcoal-900">
            Why not just paste your PDF into ChatGPT or Claude?
          </h2>
          <p className="font-sans text-sm text-charcoal-700 leading-relaxed">
            It's a fair question. When you paste a large document into a general conversational model, several invisible limitations affect accuracy, context window limits, and answer verification. Here is what makes this system fundamentally different.
          </p>
        </div>

        {/* Feature Cards Grid */}
        <div className="space-y-8 font-sans text-sm text-charcoal-900 leading-relaxed">
          {/* Item 1 */}
          <section className="space-y-2">
            <h3 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <FileCheck className="w-5 h-5 text-terracotta-600 shrink-0" />
              Your document doesn't get cut off.
            </h3>
            <p className="text-charcoal-700 leading-relaxed">
              Paste a long PDF into a chat and you often don't know how much of it the model actually read. Large documents get silently truncated, and the parts that got cut are the parts it can't answer questions about — with no warning. Here, your document is fully processed and indexed once, so every page is searchable, no matter how long it is.
            </p>
          </section>

          {/* Item 2 */}
          <section className="space-y-2">
            <h3 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-terracotta-600 shrink-0" />
              Every answer points back to a page.
            </h3>
            <p className="text-charcoal-700 leading-relaxed">
              When Claude or ChatGPT answers from a pasted document, you're trusting it on faith. Here, every claim in an answer carries an inline citation you can click — it opens the exact chunk of text and page it came from in the Source Inspector, so you can verify it yourself in seconds.
            </p>
          </section>

          {/* Item 3 */}
          <section className="space-y-2">
            <h3 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <Database className="w-5 h-5 text-terracotta-600 shrink-0" />
              Your document doesn't disappear.
            </h3>
            <p className="text-charcoal-700 leading-relaxed">
              Close a ChatGPT tab and that document context is gone — requiring you to paste it again next time. Here, once you upload something, it's indexed and stays queryable. Come back next week and ask a new question against the same document; no re-uploading needed.
            </p>
          </section>

          {/* Item 4 */}
          <section className="space-y-2">
            <h3 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <Layers className="w-5 h-5 text-terracotta-600 shrink-0" />
              Ask across documents, not just one.
            </h3>
            <p className="text-charcoal-700 leading-relaxed">
              Upload a handout, a syllabus, and your notes, and ask a question that spans all three. A single pasted-in chat can't handle cross-document synthesis once you're juggling multiple files.
            </p>
          </section>

          {/* Item 5 */}
          <section className="space-y-2">
            <h3 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-terracotta-600 shrink-0" />
              It says "I don't know" instead of guessing.
            </h3>
            <p className="text-charcoal-700 leading-relaxed">
              General-purpose chat models often fill gaps in a document with their own general knowledge, blending what's actually in your file with what the model already knew — and it's hard to tell which is which. This system only answers from what it retrieves from your documents. If it's not in there, it says so, instead of quietly making something plausible up.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
