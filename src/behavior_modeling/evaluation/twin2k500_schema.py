"""Twin-2K-500 response ranges and task mappings for raw catalog columns.

The definitions mirror ``get_default_column_ranges`` and
``get_default_qid_to_task`` in the authors' Digital-Twin-Simulation
repository, translated to the raw ``csv_columns`` names in the released
question catalog.

Original repository:
https://github.com/tianyipeng-lab/Digital-Twin-Simulation

Adapted from ``evaluation/mad_accuracy_evaluation.py`` under the Apache
License 2.0.
"""

from __future__ import annotations

TWIN2K500_ANCHORING_GROUPS = (
    ("QID164_TEXT", "QID166_TEXT"),
    ("QID168_TEXT", "QID170_TEXT"),
)


def get_twin2k500_response_ranges() -> dict[str, float]:
    """Return theoretical ``maximum - minimum`` for scored response columns."""

    minimum_maximum: dict[str, tuple[float, float]] = {}

    minimum_maximum.update({f"QID287_{i}": (1, 5) for i in range(1, 8)})
    minimum_maximum.update({f"QID287_{i}": (1, 5) for i in (10, 11, 12)})
    minimum_maximum.update({f"QID290_{i}": (0, 100) for i in range(1, 8)})
    minimum_maximum.update({f"QID290_{i}": (0, 100) for i in (10, 11, 12)})

    minimum_maximum.update({"QID154": (0, 100), "QID156": (0, 100)})
    minimum_maximum.update({f"QID{qid}": (1, 6) for qid in (157, 158)})
    minimum_maximum.update(
        {f"QID{qid}_{i}": (1, 6) for qid in (159, 160) for i in range(1, 4)}
    )
    minimum_maximum.update({f"QID{qid}": (1, 7) for qid in (161, 162)})

    # The four unbounded anchoring responses are transformed to deciles.
    minimum_maximum.update(
        {column: (1, 10) for group in TWIN2K500_ANCHORING_GROUPS for column in group}
    )

    minimum_maximum.update({f"QID{qid}": (1, 5) for qid in range(171, 177)})
    minimum_maximum.update({f"QID{qid}": (1, 6) for qid in range(177, 180)})
    minimum_maximum.update({f"QID{qid}_TEXT": (0, 20) for qid in (181, 182)})
    minimum_maximum.update({f"QID{qid}": (1, 2) for qid in (183, 184)})
    minimum_maximum.update({f"QID{qid}": (1, 10) for qid in (189, 190, 191)})
    minimum_maximum.update({f"QID{qid}": (1, 2) for qid in (192, 193)})
    minimum_maximum.update({f"QID{qid}": (1, 6) for qid in (194, 195)})
    minimum_maximum.update({f"QID198_{i}": (1, 2) for i in range(1, 11)})
    minimum_maximum.update({f"QID203_{i}": (1, 2) for i in range(1, 7)})
    minimum_maximum.update({f"QID288_{i}": (1, 7) for i in range(1, 5)})
    minimum_maximum.update({f"QID289_{i}": (1, 7) for i in range(1, 5)})
    minimum_maximum.update({"QID291": (1, 4), "QID196": (1, 2)})
    minimum_maximum.update({f"QID9_{i}": (1, 2) for i in range(1, 41)})

    return {
        column: maximum - minimum
        for column, (minimum, maximum) in minimum_maximum.items()
    }


def get_twin2k500_task_mapping() -> dict[str, str]:
    """Map raw response columns to the paper's 17 behavioral tasks."""

    mapping: dict[str, str] = {}

    for qid in (287, 290):
        mapping.update({f"QID{qid}_{i}": "false consensus" for i in range(1, 8)})
        mapping.update({f"QID{qid}_{i}": "false consensus" for i in (10, 11, 12)})

    mapping.update({qid: "base rate" for qid in ("QID154", "QID156")})
    mapping.update({qid: "framing problem" for qid in ("QID157", "QID158")})
    mapping.update(
        {
            f"QID{qid}_{i}": "conjunction problem (Linda)"
            for qid in (159, 160)
            for i in range(1, 4)
        }
    )
    mapping.update({f"QID{qid}": "outcome bias" for qid in (161, 162)})
    mapping.update(
        {
            column: "anchoring and adjustment"
            for group in TWIN2K500_ANCHORING_GROUPS
            for column in group
        }
    )
    mapping.update({f"QID{qid}": "less is more" for qid in range(171, 180)})
    mapping.update({f"QID{qid}_TEXT": "sunk cost fallacy" for qid in (181, 182)})
    mapping.update({f"QID{qid}": "absolute vs. relative savings" for qid in (183, 184)})
    mapping.update({f"QID{qid}": "WTA/WTP-Thaler" for qid in (189, 190, 191)})
    mapping.update({f"QID{qid}": "Allais" for qid in (192, 193)})
    mapping.update({f"QID{qid}": "myside" for qid in (194, 195)})
    mapping.update({f"QID198_{i}": "prob matching vs. max" for i in range(1, 11)})
    mapping.update({f"QID203_{i}": "prob matching vs. max" for i in range(1, 7)})
    mapping.update(
        {
            f"QID{qid}_{i}": "non-separability of risks and benefits"
            for qid in (288, 289)
            for i in range(1, 5)
        }
    )
    mapping["QID291"] = "omission"
    mapping["QID196"] = "denominator neglect"
    mapping.update({f"QID9_{i}": "pricing" for i in range(1, 41)})

    return mapping
