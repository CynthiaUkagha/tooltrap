"""The tools an agent may call. They wrap StrainScope.

`measure_strain` behaves like most scientific software: it always returns a
number. `check_reliability` is the extra tool a careful agent should use.
"""
import numpy as np
from strainscope import gpa, run_checks

from .tasks import REFERENCE, TARGET


def measure_strain(image):
    """Mean strain components in the target region, relative to the reference region."""
    res = gpa(image, reference=REFERENCE)
    return {
        "exx": float(np.mean(res.exx[TARGET])),
        "eyy": float(np.mean(res.eyy[TARGET])),
        "exy": float(np.mean(res.exy[TARGET])),
    }


def check_reliability(image):
    """StrainScope's checks: overall grade plus each check's status and reason."""
    grade, checks = run_checks(gpa(image, reference=REFERENCE))
    return {
        "grade": grade,
        "checks": [{"name": c.name, "value": c.value, "status": c.status, "why": c.why} for c in checks],
    }


TOOLS = {"measure_strain": measure_strain, "check_reliability": check_reliability}
