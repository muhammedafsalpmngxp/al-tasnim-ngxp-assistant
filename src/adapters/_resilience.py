"""Async resilience helper: timeout + scoped retries with backoff (transient only)."""
import asyncio
import logging
from typing import Awaitable, Callable, Tuple, Type

logger = logging.getLogger("orchestrator.resilience")


async def with_timeout(
    coro_factory: Callable[[], Awaitable],
    timeout: float,
    retries: int = 0,
    backoff: float = 1.0,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(coro_factory(), timeout=timeout)
        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning("call timed out after %ss (attempt %d/%d)", timeout, attempt + 1, retries + 1)
        except retry_on as e:
            last_exc = e
            logger.warning("transient error (attempt %d/%d): %s", attempt + 1, retries + 1, e)
        if attempt < retries:
            await asyncio.sleep(backoff * (2 ** attempt))
    raise last_exc
