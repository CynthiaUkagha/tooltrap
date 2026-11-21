"""Task generator.

Every task is a lattice image plus a question. The ground truth is known
exactly because the image is simulated from a known displacement field.
Some tasks have a numeric answer; in others the only correct response is to
abstain (the image cannot support any strain value) or to flag a defect.
"""
from dataclasses import dataclass, field

import numpy as np
from strainscope import simulate

SHAPE = (256, 256)
REFERENCE = (slice(40, 216), slice(30, 90))     # left side: the unstrained "substrate"
TARGET = (slice(40, 216), slice(160, 216))      # right side: the region to measure
QUESTION = ("This is an atomic-resolution image of a crystal film. Treat the left side "
            "(columns 30-90) as unstrained. What is the strain exx on the right side "
            "(columns 160-216)? If the image cannot support a trustworthy value, say so.")

ANSWER, ABSTAIN, FLAG, INVALID = "answer", "abstain", "flag", "invalid"

CATEGORIES = {
    # name: (expected decision, description)
    "clean":        (ANSWER,  "Clear lattice, strain step"),
    "noisy_ok":     (ANSWER,  "Noisy but still measurable lattice"),
    "no_lattice":   (ABSTAIN, "Pure noise: there is no lattice to measure"),
    "too_noisy":    (ABSTAIN, "Lattice buried in noise"),
    "dislocation":  (FLAG,    "A dislocation core sits in the measured region"),
}


@dataclass
class Task:
    task_id: str
    category: str
    image: np.ndarray
    question: str
    expected: str
    value: float = None
    tolerance: float = 0.003
    meta: dict = field(default_factory=dict)


def make_task(category, seed):
    rng = np.random.default_rng(seed)
    eps = float(rng.choice([-1, 1]) * rng.uniform(0.005, 0.03))
    rot = float(rng.uniform(0, 45))
    a = float(rng.uniform(7, 10))
    disp, _ = simulate.strain_step(SHAPE, eps_xx=eps)
    expected, _ = CATEGORIES[category]
    value = None

    if category == "clean":
        img, *_ = simulate.lattice_image(SHAPE, a=a, rotation_deg=rot, displacement=disp, noise=0.2, seed=seed)
        value = eps
    elif category == "noisy_ok":
        img, *_ = simulate.lattice_image(SHAPE, a=a, rotation_deg=rot, displacement=disp, noise=1.2, seed=seed)
        value = eps
    elif category == "no_lattice":
        img = rng.standard_normal(SHAPE)
    elif category == "too_noisy":
        img, *_ = simulate.lattice_image(SHAPE, a=a, rotation_deg=rot, displacement=disp, noise=8.0, seed=seed)
    elif category == "dislocation":
        b = simulate.rotate((a, 0.0), rot)
        cx, cy = float(rng.uniform(170, 205)), float(rng.uniform(90, 166))
        ux, uy = simulate.edge_dislocation(SHAPE, center=(cx, cy), burgers=b)
        img, *_ = simulate.lattice_image(SHAPE, a=a, rotation_deg=rot,
                                         displacement=(disp[0] + ux, disp[1] + uy), noise=0.3, seed=seed)
    else:
        raise ValueError(category)

    return Task(task_id=f"{category}-{seed:03d}", category=category, image=img,
                question=QUESTION, expected=expected, value=value,
                meta={"lattice_spacing_px": a, "rotation_deg": rot})


def make_suite(n_per_category=20, seed=0):
    tasks = []
    for k, cat in enumerate(CATEGORIES):
        for i in range(n_per_category):
            tasks.append(make_task(cat, seed * 10_000 + k * 1_000 + i))
    return tasks
