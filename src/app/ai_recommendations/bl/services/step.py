from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Concatenate, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from enum import Enum


class StepService[StatusT: Enum](Protocol):
    """A service whose methods run in a fixed order guarded by its status."""

    status: StatusT

    def _check_status(self, expected: StatusT) -> None: ...


def step[ServiceT: StepService[Any], **P, R](
    expected: Enum,
    done: Enum,
    failed: Enum,
) -> Callable[
    [Callable[Concatenate[ServiceT, P], R]],
    Callable[Concatenate[ServiceT, P], R],
]:
    """Run a service step only in the ``expected`` status, then move to ``done``.

    Any exception moves the service to ``failed``, so a half-done service is
    never reused; the facade builds a new one on the next attempt.
    """

    def decorate(
        method: Callable[Concatenate[ServiceT, P], R],
    ) -> Callable[Concatenate[ServiceT, P], R]:
        @functools.wraps(method)
        def run(service: ServiceT, *args: P.args, **kwargs: P.kwargs) -> R:
            # A wrong order is a bug in the caller, the status stays as it was
            service._check_status(expected)
            try:
                result = method(service, *args, **kwargs)
            except BaseException:
                service.status = failed
                raise
            service.status = done
            return result

        return run

    return decorate
