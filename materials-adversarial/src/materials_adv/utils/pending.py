"""Explicit markers for work that is blocked, so it can never be mistaken for working code.

The failure mode this guards against: code that runs and returns a plausible but
meaningless answer. A stub that raises cannot silently contaminate a result.
"""

from __future__ import annotations


class PendingImplementation(NotImplementedError):
    """Raised by entry points that are blocked on an external prerequisite.

    Carries *why* it is blocked and *what would unblock it*, so the traceback is
    actionable rather than a bare NotImplementedError.
    """

    def __init__(self, what: str, blocked_on: str, unblocks_when: str) -> None:
        self.what = what
        self.blocked_on = blocked_on
        self.unblocks_when = unblocks_when
        super().__init__(
            f"PENDING({blocked_on}): {what}\n"
            f"  Unblocks when: {unblocks_when}"
        )
