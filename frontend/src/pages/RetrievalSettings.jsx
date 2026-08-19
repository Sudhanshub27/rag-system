import React from 'react';
import { Sliders, Layers, Search, Cpu, CheckSquare } from 'lucide-react';

export default function RetrievalSettings() {
  return (
    <div className="w-full h-full overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans px-4 sm:px-8 py-6 sm:py-10">
      <div className="max-w-3xl mx-auto space-y-6 sm:space-y-10 pb-16">
        {/* Header */}
        <div className="border-b border-warmborder pb-5 sm:pb-6 space-y-2">
          <div className="flex items-center gap-2 text-terracotta-600 font-semibold text-xs uppercase tracking-wider font-sans">
            <Sliders className="w-4 h-4 shrink-0" /> Information Retrieval Guide
          </div>
          <h1 className="font-serif font-bold text-2xl sm:text-3xl text-charcoal-900 tracking-tight">
            Retrieval Settings, Explained
          </h1>
          <p className="font-serif italic text-charcoal-700 text-sm sm:text-base">
            These toggles change how the system searches your documents before answering.
          </p>
        </div>

        <p className="text-sm text-charcoal-700 leading-relaxed font-sans">
          None of these toggles are required — the default configuration works well for most questions. Here is what each feature does, why it helps, and how query intent (Narrow vs. Broad) is handled automatically.
        </p>

        {/* Feature 0: Dual Query Routing */}
        <div className="space-y-4 p-6 rounded-xl bg-parchment-50 border border-warmborder shadow-sm">
          <h2 className="font-serif font-bold text-xl text-charcoal-900 flex items-center gap-2">
            <Sliders className="w-5 h-5 text-terracotta-600" />
            Automatic Dual Query Routing (Narrow vs. Broad)
          </h2>
          <div className="space-y-2 text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>How it works:</strong> When you enter a question, the system's <code>QueryRouter</code> automatically classifies your intent using keyword pattern matching and BM25 score-shape heuristics:</p>
            <ul className="list-disc pl-5 space-y-1.5 text-xs text-charcoal-700">
              <li><strong>Narrow Intent (Specific Lookups):</strong> Questions asking for exact facts, dates, codes, or specific sections route through Hybrid Vector + BM25 search and Reranking.</li>
              <li><strong>Broad Intent (Document Summarization):</strong> Queries like <em>"summarize this document"</em> or <em>"explain the whole PDF"</em> route to a specialized Map-Reduce summarizer. The resulting summary is cached on disk keyed by your Tenant ID and document content SHA-256 hash.</li>
            </ul>
          </div>
        </div>

        {/* Feature 1: Multi-Query */}
        <div className="space-y-4 p-6 rounded-xl bg-parchment-50 border border-warmborder shadow-sm">
          <h2 className="font-serif font-bold text-xl text-charcoal-900 flex items-center gap-2">
            <Layers className="w-5 h-5 text-terracotta-600" />
            1. Multi-Query Expansion
          </h2>
          <div className="space-y-2 text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>What it is:</strong> Instead of searching with only your exact wording, the system rephrases your question two or three different ways in the background, then searches for all of them.</p>
            <p><strong>Why it helps:</strong> If your question is vague, oddly worded, or uses different terms than the document does, a single search can miss the relevant passage entirely. Multiple phrasings widen the net.</p>
            <p><strong>When to use it:</strong> Turn it on for broad or loosely-worded questions ("what's this about," "explain the process"). Leave it off for precise lookups where you already know the exact term used in the document — it adds a small amount of extra time for little benefit there.</p>
            <div className="p-3 bg-terracotta-100/50 border-l-2 border-terracotta-600 rounded text-xs font-serif italic text-charcoal-900">
              <strong>What changes in the answer:</strong> You'll typically get more complete coverage of a topic, since more relevant passages get pulled in — at the cost of a slightly slower response.
            </div>
          </div>
        </div>

        {/* Feature 2: HyDE */}
        <div className="space-y-4 p-6 rounded-xl bg-parchment-50 border border-warmborder shadow-sm">
          <h2 className="font-serif font-bold text-xl text-charcoal-900 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-terracotta-600" />
            2. HyDE (Hypothetical Document Embeddings)
          </h2>
          <div className="space-y-2 text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>What it is:</strong> Before searching, the system asks the model to write a short hypothetical answer to your question, then searches for real chunks that <em>resemble</em> that hypothetical answer — rather than searching directly on your question's wording.</p>
            <p><strong>Why it helps:</strong> Questions and answers are often phrased very differently (a question asks "why," a passage explains without ever using the word "why"). Searching by what an answer would <em>look like</em> often finds better matches than searching by the question itself.</p>
            <p><strong>When to use it:</strong> Useful for conceptual or "why/how" questions where the document's phrasing likely differs from how you'd naturally ask. Less useful for simple fact lookups ("what is the deadline mentioned").</p>
            <div className="p-3 bg-terracotta-100/50 border-l-2 border-terracotta-600 rounded text-xs font-serif italic text-charcoal-900">
              <strong>What changes in the answer:</strong> Often improves relevance on interpretive questions; adds one extra generation step, so responses take slightly longer.
            </div>
          </div>
        </div>

        {/* Feature 3: Hybrid Search */}
        <div className="space-y-4 p-6 rounded-xl bg-parchment-50 border border-warmborder shadow-sm">
          <h2 className="font-serif font-bold text-xl text-charcoal-900 flex items-center gap-2">
            <Search className="w-5 h-5 text-terracotta-600" />
            3. Hybrid Search (Keyword BM25 + Vector Search)
          </h2>
          <div className="space-y-2 text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>What it is:</strong> Combines two search methods — BM25 keyword matching (finds exact terms, numbers, names) and semantic vector search (finds passages that mean the same thing even with different words) — then merges the results using Reciprocal Rank Fusion.</p>
            <p><strong>Why it helps:</strong> Vector search alone can miss exact terms, dates, or codes because it focuses on meaning over exact wording. Keyword search alone misses paraphrased content. Together they cover both.</p>
            <p><strong>When to use it:</strong> Keep this on by default — it's rarely worse than either method alone. It matters most when your question includes specific names, numbers, or exact phrases from the document.</p>
            <div className="p-3 bg-terracotta-100/50 border-l-2 border-terracotta-600 rounded text-xs font-serif italic text-charcoal-900">
              <strong>What changes in the answer:</strong> More reliable retrieval overall, especially for questions mixing a specific term with a general concept ("what does section 4.2 say about deadlines").
            </div>
          </div>
        </div>

        {/* Feature 4: Reranking */}
        <div className="space-y-4 p-6 rounded-xl bg-parchment-50 border border-warmborder shadow-sm">
          <h2 className="font-serif font-bold text-xl text-charcoal-900 flex items-center gap-2">
            <CheckSquare className="w-5 h-5 text-terracotta-600" />
            4. Cross-Encoder Reranking
          </h2>
          <div className="space-y-2 text-sm text-charcoal-900 font-sans leading-relaxed">
            <p><strong>What it is:</strong> After the initial search returns a batch of candidate passages, a second, more careful cross-encoder model re-scores them for relevance to your specific question and keeps only the best ones before generating an answer.</p>
            <p><strong>Why it helps:</strong> Initial search is fast but approximate — it can rank a loosely-related passage above a more relevant one. Reranking is slower but more precise, and cleans that up.</p>
            <p><strong>When to use it:</strong> Leave it on for most questions — it meaningfully improves answer quality. You might turn it off only if you're testing raw retrieval quality or want the fastest possible response.</p>
            <div className="p-3 bg-terracotta-100/50 border-l-2 border-terracotta-600 rounded text-xs font-serif italic text-charcoal-900">
              <strong>What changes in the answer:</strong> Answers tend to be more tightly grounded in the most relevant chunks, with less "close but not quite right" content, at a small added cost in response time.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
