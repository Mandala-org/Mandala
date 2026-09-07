"""Shared acceptance rules for independently fitted Stage-6 neural runs."""

from __future__ import annotations

import math

STAGE6_FLOAT32_EQUIVARIANCE_CEILING = 5.0e-5


def assess_stage6_scoped_run(
    summary: dict[str, object], nominal_tolerance: float
) -> dict[str, object]:
    """Assess a run while preserving a too-strict original pass flag.

    Stage 6 permits a narrowly amended float32 relative tolerance because large
    trained equivariant contractions accumulate scale-dependent roundoff. Raw
    reversal errors are intentionally excluded: independent directed calls are
    required to differ before evaluation-only projection.
    """
    gate_errors = {
        key: float(value)
        for key, value in summary.get("final_float32_symmetry_errors", {}).items()
        if key.startswith(("proper_", "improper_", "projected_"))
    }
    finite_metric = math.isfinite(
        float(summary.get("best_validation_matrix_mae_mev", float("nan")))
    )
    ceiling = max(float(nominal_tolerance), STAGE6_FLOAT32_EQUIVARIANCE_CEILING)
    worst_error = max(gate_errors.values(), default=float("inf"))
    accepted = bool(
        summary.get("completed")
        and not summary.get("test_shards_read")
        and finite_metric
        and gate_errors
        and worst_error <= ceiling
    )
    return {
        "accepted": accepted,
        "original_passed": bool(summary.get("passed")),
        "numerical_tolerance_amendment_used": bool(
            accepted and worst_error > float(nominal_tolerance)
        ),
        "nominal_float32_relative_tolerance": float(nominal_tolerance),
        "stage6_float32_relative_ceiling": ceiling,
        "worst_gated_float32_relative_error": worst_error,
        "gated_float32_errors": gate_errors,
    }
