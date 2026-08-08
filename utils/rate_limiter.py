"""
Per-User Rate Limiter for RAG Pipeline API protection.
Enforces request rate limits and quotas per user_id to prevent API key exhaustion.
"""

import time
from collections import defaultdict

from utils.logger import logger


class RateLimitExceededError(Exception):
    """Raised when a user exceeds their query rate limit or quota."""

    pass


class RateLimiter:
    """
    Sliding window rate limiter to restrict query frequency per user_id.

    Args:
        max_requests_per_minute: Maximum queries allowed per minute per user (default: 6).
        max_requests_per_hour: Maximum queries allowed per hour per user (default: 50).
    """

    def __init__(
        self,
        max_requests_per_minute: int = 6,
        max_requests_per_hour: int = 50,
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self._user_timestamps: dict[str, list[float]] = defaultdict(list)

    def check_rate_limit(self, user_id: str) -> None:
        """
        Check if user_id is within rate limit thresholds.
        Removes timestamps older than 1 hour and counts remaining requests.

        Raises:
            RateLimitExceededError if limit is breached.
        """
        now = time.time()
        timestamps = self._user_timestamps[user_id]

        # Prune entries older than 3600 seconds (1 hour)
        cutoff_hour = now - 3600
        timestamps = [ts for ts in timestamps if ts > cutoff_hour]
        self._user_timestamps[user_id] = timestamps

        # Check 1-hour window quota
        if len(timestamps) >= self.max_requests_per_hour:
            logger.warning(
                f"Rate limit exceeded for user_id='{user_id}' (1-hour quota: {self.max_requests_per_hour})."
            )
            raise RateLimitExceededError(
                f"Hourly query limit reached ({self.max_requests_per_hour} queries/hr). "
                "Please wait before making additional requests."
            )

        # Check 1-minute window quota
        cutoff_minute = now - 60
        recent_minute_count = sum(1 for ts in timestamps if ts > cutoff_minute)
        if recent_minute_count >= self.max_requests_per_minute:
            logger.warning(
                f"Rate limit exceeded for user_id='{user_id}' (1-min limit: {self.max_requests_per_minute})."
            )
            raise RateLimitExceededError(
                f"Rate limit exceeded ({self.max_requests_per_minute} queries/min). "
                "Please wait a few seconds before asking your next question."
            )

        # Record this request
        timestamps.append(now)

    def get_user_usage(self, user_id: str) -> dict[str, int]:
        """Return request metrics for a user."""
        now = time.time()
        timestamps = [ts for ts in self._user_timestamps[user_id] if ts > now - 3600]
        recent_minute = sum(1 for ts in timestamps if ts > now - 60)
        return {
            "requests_last_minute": recent_minute,
            "requests_last_hour": len(timestamps),
            "max_per_minute": self.max_requests_per_minute,
            "max_per_hour": self.max_requests_per_hour,
        }


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests_per_minute=6, max_requests_per_hour=50)
