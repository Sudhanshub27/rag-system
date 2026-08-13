"""
Cache Cleanup Utilities
Maintains disk usage bounds by evicting stale cache files and enforcing total size caps.
"""

import time
from pathlib import Path

from utils.logger import logger

DEFAULT_SUMMARY_CACHE = Path(".cache/summaries")


def clean_summary_cache(
    cache_dir: Path | str = DEFAULT_SUMMARY_CACHE,
    max_age_days: int = 30,
    max_size_mb: int = 50,
) -> int:
    """
    Clean summary cache entries exceeding age or size limits.

    Args:
        cache_dir: Path to summary cache directory.
        max_age_days: Evict entries older than N days.
        max_size_mb: Maximum total size in megabytes for the cache directory.

    Returns:
        Number of evicted cache files.
    """
    target_dir = Path(cache_dir)
    if not target_dir.exists():
        return 0

    evicted_count = 0
    now = time.time()
    max_age_seconds = max_age_days * 86400

    cache_files: list[tuple[Path, float, int]] = []
    total_size_bytes = 0

    for path in target_dir.glob("*.json"):
        try:
            stat = path.stat()
            age_seconds = now - stat.st_mtime
            if age_seconds > max_age_seconds:
                path.unlink()
                evicted_count += 1
                logger.info(
                    f"CacheCleanup: Deleted stale cache file '{path.name}' (age: {age_seconds / 86400:.1f} days)"
                )
            else:
                cache_files.append((path, stat.st_mtime, stat.st_size))
                total_size_bytes += stat.st_size
        except Exception as e:
            logger.warning(f"CacheCleanup error examining '{path.name}': {e}")

    # Enforce size cap (evict oldest first if size limit exceeded)
    max_bytes = max_size_mb * 1024 * 1024
    if total_size_bytes > max_bytes:
        # Sort by mtime ascending (oldest first)
        cache_files.sort(key=lambda x: x[1])
        for path, mtime, size in cache_files:
            if total_size_bytes <= max_bytes:
                break
            try:
                path.unlink()
                evicted_count += 1
                total_size_bytes -= size
                logger.info(
                    f"CacheCleanup: Size cap exceeded. Evicted '{path.name}' ({size / 1024:.1f} KB)"
                )
            except Exception as e:
                logger.warning(f"CacheCleanup error evicting '{path.name}': {e}")

    return evicted_count


if __name__ == "__main__":
    count = clean_summary_cache()
    print(f"Summary Cache Cleanup Complete. Evicted {count} file(s).")
