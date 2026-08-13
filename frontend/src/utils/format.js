/**
 * Converts Markdown formatted text into clean plain text for clipboard copying.
 * Preserves paragraphs and lists while stripping markdown syntax (#, **, *, `, [], etc.).
 */
export function markdownToPlainText(markdown) {
  if (!markdown) return '';

  return markdown
    // Remove code blocks
    .replace(/```[\s\S]*?```/g, (match) => {
      // Keep code contents without ```
      return match.replace(/^```[a-z]*\n?/i, '').replace(/\n?```$/, '');
    })
    // Remove inline code ticks
    .replace(/`([^`]+)`/g, '$1')
    // Remove headers (#, ##, ###)
    .replace(/^#{1,6}\s+/gm, '')
    // Remove bold and italic formatting (**text**, *text*, __text__, _text_)
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(\*|_)(.*?)\1/g, '$2')
    // Remove images (![alt](url))
    .replace(/!\[(.*?)\]\([^)]+\)/g, '$1')
    // Remove links ([text](url)) -> text
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
    // Convert bullet markers (* item, - item) to bullet points or clean text
    .replace(/^\s*[-*+]\s+/gm, '• ')
    // Remove blockquotes (> quote)
    .replace(/^\s*>\s+/gm, '')
    // Remove horizontal rules (---, ***, ___)
    .replace(/^[-*_]{3,}\s*$/gm, '')
    // Clean extra blank lines
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}
