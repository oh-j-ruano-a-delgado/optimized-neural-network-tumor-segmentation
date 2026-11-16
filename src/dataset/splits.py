import random
from collections import defaultdict
from typing import Dict, FrozenSet, List, Mapping, Tuple


def group_patients_by_label(patient_labels: Mapping[str, int]) -> Dict[int, List[str]]:
    patients_by_label: Dict[int, List[str]] = defaultdict(list)
    for patient_id in sorted(patient_labels):
        patients_by_label[patient_labels[patient_id]].append(patient_id)
    return patients_by_label


def compute_holdout_count(group_size: int, holdout_fraction: float) -> int:
    # Groups with two or more patients always contribute to both subsets so that
    # every tumor type stays represented on each side of the split.
    if group_size < 2:
        return 0
    return min(max(round(group_size * holdout_fraction), 1), group_size - 1)


def split_patients_stratified(
    patient_labels: Mapping[str, int], holdout_fraction: float, seed: int
) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    # Splitting by patient instead of by slice prevents adjacent slices of the same
    # tumor from leaking between subsets, which would inflate evaluation metrics.
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}.")

    random_generator = random.Random(seed)
    kept_patients: List[str] = []
    holdout_patients: List[str] = []

    for _, patient_ids in sorted(group_patients_by_label(patient_labels).items()):
        shuffled_patient_ids = list(patient_ids)
        random_generator.shuffle(shuffled_patient_ids)
        holdout_count = compute_holdout_count(len(shuffled_patient_ids), holdout_fraction)
        holdout_patients.extend(shuffled_patient_ids[:holdout_count])
        kept_patients.extend(shuffled_patient_ids[holdout_count:])

    return frozenset(kept_patients), frozenset(holdout_patients)
