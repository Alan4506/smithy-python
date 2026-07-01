# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from asyncio import gather, sleep

import pytest
from smithy_core.aio.interfaces import retries as retries_interface
from smithy_core.aio.retries import StandardRetryStrategy
from smithy_core.exceptions import CallError, ClientTimeoutError, RetryError
from smithy_core.retries import (
    ExponentialBackoffJitterType,
    ExponentialRetryBackoffStrategy,
    StandardRetryQuota,
)


# TODO: Refactor this to use a smithy-testing generated client
async def retry_operation(
    strategy: retries_interface.RetryStrategy,
    responses: list[int | Exception],
) -> tuple[str, int]:
    token = await strategy.acquire_initial_retry_token()
    response_iter = iter(responses)

    while True:
        if token.retry_delay:
            await sleep(token.retry_delay)

        response = next(response_iter)
        attempt = token.retry_count + 1

        # Success case
        if response == 200:
            await strategy.record_success(token=token)
            return "success", attempt

        # Error case - either status code or exception
        if isinstance(response, Exception):
            error = response
        else:
            error = CallError(
                fault="server" if response >= 500 else "client",
                message=f"HTTP {response}",
                is_retry_safe=response >= 500,
            )

        try:
            token = await strategy.refresh_retry_token_for_retry(
                token_to_renew=token, error=error
            )
        except RetryError:
            raise error


async def test_standard_retry_eventually_succeeds():
    quota = StandardRetryQuota(initial_capacity=500)
    strategy = StandardRetryStrategy(max_attempts=3, retry_quota=quota)

    result, attempts = await retry_operation(strategy, [500, 500, 200])

    assert result == "success"
    assert attempts == 3
    # 2 retries drain 14 each (500 -> 472); success releases the last retry's
    # cost (14) back -> 486.
    assert quota.available_capacity == 486


async def test_standard_retry_fails_due_to_max_attempts():
    quota = StandardRetryQuota(initial_capacity=500)
    strategy = StandardRetryStrategy(max_attempts=3, retry_quota=quota)

    with pytest.raises(CallError, match="502"):
        await retry_operation(strategy, [502, 502, 502])

    # 2 retries drain 14 each; no success release on max-attempts failure.
    assert quota.available_capacity == 472


async def test_retry_quota_exhausted_after_single_retry():
    quota = StandardRetryQuota(initial_capacity=14)
    strategy = StandardRetryStrategy(max_attempts=3, retry_quota=quota)

    with pytest.raises(CallError, match="502"):
        await retry_operation(strategy, [500, 502])

    assert quota.available_capacity == 0


async def test_retry_quota_prevents_retries_when_quota_zero():
    quota = StandardRetryQuota(initial_capacity=0)
    strategy = StandardRetryStrategy(max_attempts=3, retry_quota=quota)

    with pytest.raises(CallError, match="500"):
        await retry_operation(strategy, [500])

    assert quota.available_capacity == 0


async def test_retry_quota_stops_retries_when_exhausted():
    quota = StandardRetryQuota(initial_capacity=20)
    strategy = StandardRetryStrategy(max_attempts=5, retry_quota=quota)

    with pytest.raises(CallError, match="502"):
        await retry_operation(strategy, [500, 502])

    # First retry drains 14 (20 -> 6); second retry needs 14 > 6 -> exhausted.
    assert quota.available_capacity == 6


async def test_retry_quota_recovers_after_successful_responses():
    quota = StandardRetryQuota(initial_capacity=30)
    strategy = StandardRetryStrategy(max_attempts=5, retry_quota=quota)

    # First operation: 2 retries then success (30 -> 2, release 14 -> 16)
    await retry_operation(strategy, [500, 502, 200])
    assert quota.available_capacity == 16

    # Second operation: 1 retry then success (16 -> 2, release 14 -> 16)
    await retry_operation(strategy, [500, 200])
    assert quota.available_capacity == 16


async def test_retry_quota_shared_across_concurrent_operations():
    quota = StandardRetryQuota(initial_capacity=500)
    backoff = ExponentialRetryBackoffStrategy(
        backoff_scale_value=1,
        max_backoff=10,
        jitter_type=ExponentialBackoffJitterType.FULL,
    )
    strategy = StandardRetryStrategy(
        max_attempts=5,
        retry_quota=quota,
        backoff_strategy=backoff,
    )

    result1, result2 = await gather(
        retry_operation(strategy, [500, 500, 200]),
        retry_operation(strategy, [500, 200]),
    )

    assert result1 == ("success", 3)
    assert result2 == ("success", 2)
    # op1 keeps 14 consumed (28 drained, 14 released on success); op2 nets 0.
    assert quota.available_capacity == 486


async def test_retry_quota_handles_timeout_errors():
    quota = StandardRetryQuota(initial_capacity=500)
    strategy = StandardRetryStrategy(max_attempts=3, retry_quota=quota)

    timeout1 = ClientTimeoutError()
    timeout2 = ClientTimeoutError()

    result, attempts = await retry_operation(strategy, [timeout1, timeout2, 200])

    assert result == "success"
    assert attempts == 3
    # Timeouts are no longer charged a special cost; they use RETRY_COST (14).
    # 2 retries drain 28 (500 -> 472); success releases 14 -> 486.
    assert quota.available_capacity == 486
