import React from 'react';
import { Shield, Key, Lock, Trash2, UserX, ShieldCheck, Cpu } from 'lucide-react';

export default function Privacy() {
  return (
    <div className="w-full h-full overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans px-8 py-10">
      <div className="max-w-3xl mx-auto space-y-10 pb-16">
        {/* Header */}
        <div className="border-b border-warmborder pb-6 space-y-2">
          <div className="flex items-center gap-2 text-terracotta-600 font-semibold text-xs uppercase tracking-wider font-sans">
            <Shield className="w-4 h-4" /> Data Protection & Privacy Architecture
          </div>
          <h1 className="font-serif font-bold text-3xl text-charcoal-900 tracking-tight">
            Privacy & Your Data
          </h1>
          <p className="font-serif italic text-charcoal-700 text-base">
            No signup, zero identity tracking, and complete per-tenant isolation.
          </p>
        </div>

        {/* Content Sections */}
        <div className="space-y-8 font-sans text-sm text-charcoal-900 leading-relaxed">
          {/* Section 1 */}
          <section className="space-y-2 p-5 bg-parchment-50 border border-warmborder rounded-xl shadow-sm">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <UserX className="w-5 h-5 text-terracotta-600 shrink-0" />
              We don't know who you are.
            </h2>
            <p className="text-charcoal-700 leading-relaxed">
              There's no signup, no login, no email address, no name. When you first open this app, your browser is given a private, randomly generated ID — that's it. We can't connect it to you personally, and we don't try to.
            </p>
          </section>

          {/* Section 2 */}
          <section className="space-y-2 p-5 bg-parchment-50 border border-warmborder rounded-xl shadow-sm">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <Lock className="w-5 h-5 text-terracotta-600 shrink-0" />
              Your documents are yours.
            </h2>
            <p className="text-charcoal-700 leading-relaxed">
              What you upload is used for exactly one thing: answering your questions about it. It is never used to train any model, never reviewed by us, never shared with anyone else. No other user of this app can see your documents, your questions, or your answers — your data is stored in a space that's isolated to your ID alone.
            </p>
          </section>

          {/* API Data Flow Clarification */}
          <section className="space-y-3 p-5 bg-parchment-50 border border-warmborder rounded-xl shadow-sm">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-terracotta-600 shrink-0" />
              LLM API Privacy & Approved Zero-Training Providers
            </h2>
            <p className="text-charcoal-700 leading-relaxed">
              We strictly enforce provider selection to only include services whose official terms explicitly guarantee <strong>ZERO model training</strong> on submitted prompts and document chunks:
            </p>
            <ul className="list-disc pl-5 space-y-2 text-charcoal-700">
              <li><strong>Groq Free API (Llama 3.3 70B / Llama 3.1 8B):</strong> Official Terms guarantee zero data retention and zero training on API data.</li>
              <li><strong>Ollama (100% Offline Local Models):</strong> Runs Llama 3.3 or Qwen 2.5 directly on your machine. Absolutely zero data leaves your local hardware.</li>
              <li><strong>OpenAI API (GPT-4o / GPT-4o-mini):</strong> Official OpenAI API Business Terms explicitly state: <em>"OpenAI does not train models on data submitted to OpenAI API endpoints."</em></li>
              <li><strong>Anthropic Claude API:</strong> Official Commercial Terms state: <em>"Anthropic does not train models on Customer Data submitted through our API."</em></li>
              <li><strong>DeepSeek API:</strong> Official DeepSeek Developer API Terms state prompt data is not retained or used for foundation model training.</li>
            </ul>
            <div className="p-3 bg-terracotta-100/60 border-l-2 border-terracotta-600 rounded text-xs text-charcoal-900 font-sans mt-3">
              <strong>Local PII Anonymizer Protection:</strong> Before any chunk is transmitted to external cloud APIs, our local PII Scrubber redacts names, email addresses, phone numbers, and IP addresses into placeholders (`[EMAIL_1]`, `[PERSON_1]`).
            </div>
          </section>

          {/* Tenant ID Security Guarantee */}
          <section className="space-y-3 p-5 bg-parchment-50 border border-warmborder rounded-xl shadow-sm">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-terracotta-600 shrink-0" />
              Can someone steal my data if they see or copy my Tenant ID?
            </h2>
            <p className="text-charcoal-700 leading-relaxed">
              <strong>No. Your data cannot be stolen simply by someone obtaining your Tenant ID string.</strong>
            </p>
            <p className="text-charcoal-700 leading-relaxed">
              Access to your vector index and documents is bound strictly to your browser's private HTTP-Only session token. Simply viewing, copying, or sharing the Tenant ID string does not grant API access or document retrieval permissions to unauthorized third parties. Furthermore, document chunks are stored as mathematical embeddings and cannot be mass-downloaded or scraped.
            </p>
          </section>

          {/* Section 3 */}
          <section className="space-y-3 p-5 bg-parchment-50 border border-warmborder rounded-xl shadow-sm">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2">
              <Key className="w-5 h-5 text-terracotta-600 shrink-0" />
              How your data survives without an account
            </h2>
            <p className="text-charcoal-700 leading-relaxed">
              This is the part that usually needs explaining: if there's no login, how does the app remember you next week?
            </p>
            <p className="text-charcoal-700 leading-relaxed">
              Your browser holds a small, private key — an HTTP cookie — the first time you visit. That key is what "recognizes" you on your next visit; it's not tied to your name or identity, just to that browser. As long as that cookie is there, your uploaded documents and chat history are exactly where you left them.
            </p>
            <div className="p-4 rounded-lg bg-terracotta-100/60 border border-terracotta-600/20 text-xs text-charcoal-900 leading-relaxed">
              <strong>The honest trade-off:</strong> If you clear your cookies, use a private/incognito window, or switch to a different browser or device, that key is gone — and so is access to that data. There's no account to log back into and recover it, because there's no account at all. This isn't a bug; it's the deliberate cost of not asking you to sign up.
            </div>
          </section>

          {/* Section 4 */}
          <section className="space-y-2 p-5 bg-parchment-50 border border-warmborder rounded-xl shadow-sm">
            <h2 className="font-serif font-bold text-lg text-rust-600 flex items-center gap-2">
              <Trash2 className="w-5 h-5 text-rust-600 shrink-0" />
              Deleting your data
            </h2>
            <p className="text-charcoal-700 leading-relaxed">
              The <strong>"Delete all my data"</strong> button in the sidebar does exactly what it says: it permanently and immediately removes every document and chunk associated with your ID from our vector store. There's no recovery after that — treat it as final.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
