"""
Unit tests for Per-User Rate Limiter module.
"""

import pytest

from utils.rate_limiter import RateLimiter, RateLimitExceededError


def test_rate_limiter_allows_under_limit():
    limiter = RateLimiter(max_requests_per_minute=3, max_requests_per_hour=10)
    limiter.check_rate_limit("user_a")
    limiter.check_rate_limit("user_a")
    usage = limiter.get_user_usage("user_a")
    assert usage["requests_last_minute"] == 2
    assert usage["requests_last_hour"] == 2


def test_rate_limiter_blocks_over_limit():
    limiter = RateLimiter(max_requests_per_minute=2, max_requests_per_hour=10)
    limiter.check_rate_limit("user_b")
    limiter.check_rate_limit("user_b")
    with pytest.raises(RateLimitExceededError):
        limiter.check_rate_limit("user_b")


def test_rate_limiter_isolated_per_user():
    limiter = RateLimiter(max_requests_per_minute=1, max_requests_per_hour=10)
    limiter.check_rate_limit("user_c")
    # user_d should not be blocked by user_c
    limiter.check_rate_limit("user_d")
