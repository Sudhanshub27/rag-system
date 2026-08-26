import React from 'react';
import { Link } from 'react-router-dom';
import {
  Shield,
  Key,
  Lock,
  Trash2,
  UserX,
  ShieldCheck,
  Cpu,
  ArrowRight,
  ArrowLeft,
} from 'lucide-react';
import LandingFooter from '../components/landing/LandingFooter';

export default function Privacy() {
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
              <Shield className="w-4 h-4 text-terracotta-600" />
              Data Protection & Privacy Architecture
            </span>
          </div>

          <div className="space-y-2">
            <h1 className="font-serif font-bold text-3xl sm:text-4xl text-charcoal-900 tracking-tight">
              Privacy & Your Data
            </h1>
            <p className="font-serif italic text-charcoal-700 text-base sm:text-lg">
              No signup, zero identity tracking, and complete per-tenant isolation.
            </p>
          </div>
        </div>

        {/* Content Sections */}
        <div className="space-y-8 font-sans text-sm text-charcoal-900 leading-relaxed">
          {/* Section 1 */}
          <section className="space-y-3 p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                <UserX className="w-4 h-4" />
              </div>
              <span>We don't know who you are.</span>
            </h2>
            <p className="text-charcoal-700 leading-relaxed pl-0 sm:pl-10">
              There's no signup, no login, no email address, no name. When you first open this app, your browser is given a private, randomly generated ID — that's it. We can't connect it to you personally, and we don't try to.
            </p>
          </section>

          {/* Section 2 */}
          <section className="space-y-3 p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                <Lock className="w-4 h-4" />
              </div>
              <span>Multi-Tenant Data Container & Summary Cache Isolation</span>
            </h2>
            <p className="text-charcoal-700 leading-relaxed pl-0 sm:pl-10">
              What you upload is used for exactly one thing: answering your questions about it. Every tenant receives a dedicated, physically separate vector database container (<code className="text-xs bg-parchment-200 px-1 py-0.5 rounded font-mono font-semibold border border-warmborder">tenant_&lt;tenant_id&gt;</code>) in ChromaDB, as well as isolated document summary caches (<code className="text-xs bg-parchment-200 px-1 py-0.5 rounded font-mono font-semibold border border-warmborder">summary_&lt;tenant_id&gt;_&lt;hash&gt;.json</code>). No other user or session can retrieve, search, or view your vector embeddings, BM25 indices, or cached summaries.
            </p>
          </section>

          {/* API Data Flow Clarification */}
          <section className="space-y-4 p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                <Cpu className="w-4 h-4" />
              </div>
              <span>LLM API Privacy & Built-in Local PII Parsing</span>
            </h2>
            <div className="space-y-3 pl-0 sm:pl-10">
              <p className="text-charcoal-700 leading-relaxed">
                Privacy is enforced by default at the architectural level before any query or document chunk is transmitted:
              </p>
              <ul className="list-disc pl-5 space-y-2 text-charcoal-700">
                <li><strong>Default Groq Free API (Llama 3.3 70B):</strong> Official Groq API Terms explicitly guarantee zero data retention and <strong>zero training</strong> on your API prompts or document text.</li>
                <li><strong>Built-in Local PII Parsing:</strong> Personal identifiers (names, email addresses, phone numbers, and IP addresses) are automatically parsed and redacted into placeholders (`[EMAIL_1]`, `[PERSON_1]`) locally on your device before payload transmission.</li>
                <li><strong>Bring Your Own Key (BYOK):</strong> Supports pasting your own keys for major enterprise API providers (OpenAI, Anthropic Claude, DeepSeek, Google Gemini, OpenRouter) with complete per-tenant isolation.</li>
              </ul>
              <div className="p-3 bg-terracotta-100/60 border-l-2 border-terracotta-600 rounded text-xs text-charcoal-900 font-sans mt-3">
                <strong>Local PII Protection:</strong> Local PII parsing can be enabled on all requests. Personal identifiers like names and emails are masked before transmission to external model providers.
              </div>
            </div>
          </section>

          {/* Tenant ID Security Guarantee */}
          <section className="space-y-3 p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <span>Can someone steal my data if they see or copy my Tenant ID?</span>
            </h2>
            <div className="space-y-2 pl-0 sm:pl-10 text-charcoal-700 leading-relaxed">
              <p>
                <strong>No. Your data cannot be accessed simply by someone obtaining your Tenant ID string.</strong>
              </p>
              <p>
                Access to your vector index and documents is bound strictly to your browser's private HTTP-Only session token. Simply viewing, copying, or sharing the Tenant ID string does not grant API access or document retrieval permissions to unauthorized third parties. Furthermore, document chunks are stored as mathematical embeddings and cannot be mass-downloaded or scraped.
              </p>
            </div>
          </section>

          {/* Section 3 */}
          <section className="space-y-3 p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                <Key className="w-4 h-4" />
              </div>
              <span>How your data survives without an account</span>
            </h2>
            <div className="space-y-3 pl-0 sm:pl-10 text-charcoal-700 leading-relaxed">
              <p>
                If there's no login, how does the app remember you next time?
              </p>
              <p>
                Your browser holds a small, private key — an HTTP cookie — the first time you visit. That key is what "recognizes" your session on your next visit; it's not tied to your name or identity, just to that browser. As long as that cookie is present, your uploaded documents and workspace history remain accessible.
              </p>
              <div className="p-4 rounded-xl bg-parchment-200/80 border border-warmborder text-xs text-charcoal-900 leading-relaxed">
                <strong>Session Trade-off:</strong> If you clear your cookies, use a private/incognito window, or switch to a different browser or device, that key is reset — and so is access to that data. There is no account recovery process because no user account was ever registered.
              </div>
            </div>
          </section>

          {/* Section 4 */}
          <section className="space-y-3 p-6 bg-parchment-50 border border-warmborder rounded-2xl shadow-2xs">
            <h2 className="font-serif font-bold text-lg text-charcoal-900 flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-terracotta-100/70 border border-terracotta-600/20 flex items-center justify-center text-terracotta-700 shrink-0">
                <Trash2 className="w-4 h-4 text-terracotta-700" />
              </div>
              <span>Deleting your data</span>
            </h2>
            <p className="text-charcoal-700 leading-relaxed pl-0 sm:pl-10">
              The <strong>"Delete all my data"</strong> action in the workspace sidebar permanently removes every document, vector embedding, and index associated with your session ID from the vector store immediately.
            </p>
          </section>
        </div>

        {/* Bottom CTA Block */}
        <div className="border-t border-warmborder pt-10 pb-4">
          <div className="p-8 sm:p-10 rounded-3xl bg-parchment-200/60 border border-warmborder text-center space-y-4 shadow-xs">
            <h2 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900">
              Start with your documents
            </h2>
            <p className="font-sans text-xs sm:text-sm text-charcoal-700 max-w-md mx-auto">
              Private, account-free document Q&A in your browser.
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
