"""Dataset loading. PENDING(dataset).

Blocked because the OpenPoly schema is UNVERIFIED. The paper (Wang et al., CJPS
2025, DOI 10.1007/s10118-025-3402-y) is paywalled, so column names, file layout
and Tg units must come from the files themselves.

Guessing a column name here would be the exact failure this project forbids: the
code would run, silently select the wrong column, and produce numbers that look
plausible. Run scripts/audit_dataset.py first; it DISCOVERS the schema and
proposes candidate columns for human confirmation.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.pending import PendingImplementation

_UNBLOCKS = (
    "scripts/audit_dataset.py has been run on the real OpenPoly files and the "
    "representation column, Tg column and Tg units are recorded in configs/dataset.yaml"
)


def load_raw(path: str | Path):
    raise PendingImplementation(
        what="load_raw(): reading the OpenPoly table requires its verified schema",
        blocked_on="dataset",
        unblocks_when=_UNBLOCKS,
    )


def load_property_subset(path: str | Path, property_column: str):
    raise PendingImplementation(
        what=(
            "load_property_subset(): the Tg column name and its non-null count are "
            "unverified. OpenPoly spreads ~3985 pairs over 26 properties, so the Tg "
            "slice size must be MEASURED, not assumed."
        ),
        blocked_on="dataset",
        unblocks_when=_UNBLOCKS,
    )
