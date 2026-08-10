/**
 * Deduplicates retrieved chunks into unique (source_file, page_number) entries.
 * Layer: Frontend Presentation Helper.
 * Replaces multiple raw chunk items with a single consolidated reference entry
 * per unique (source_file, page_number), sorted by page number ascending.
 */
export function deduplicateCitations(retrievedChunks = []) {
  if (!retrievedChunks || !Array.isArray(retrievedChunks) || retrievedChunks.length === 0) {
    return [];
  }

  const map = new Map();

  retrievedChunks.forEach((rc) => {
    if (!rc) return;
    const rawSource = rc.source || 'Document';
    const filename = rawSource.split('/').pop();
    const pageNum = rc.page !== undefined && rc.page !== null ? rc.page : 1;
    const key = `${filename}::${pageNum}`;

    if (!map.has(key)) {
      map.set(key, {
        id: map.size + 1,
        source: filename,
        fullPath: rawSource,
        page: pageNum,
        chunks: rc.text ? [rc.text] : [],
        score: rc.score || 1.0,
      });
    } else {
      const existing = map.get(key);
      if (rc.text && !existing.chunks.includes(rc.text)) {
        existing.chunks.push(rc.text);
      }
      if (rc.score && rc.score > existing.score) {
        existing.score = rc.score;
      }
    }
  });

  return Array.from(map.values()).sort((a, b) => a.page - b.page);
}

/**
 * Finds a citation by number or ID for inline popovers.
 */
export function findCitationById(citations = [], id) {
  if (!citations || !Array.isArray(citations)) return null;
  const numId = parseInt(id, 10);
  return citations.find((c) => c.id === numId) || citations[numId - 1] || null;
}
