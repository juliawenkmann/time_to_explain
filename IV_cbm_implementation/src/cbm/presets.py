"""Preset and dataset-profile configuration for the CBM notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetProfile:
    candidates: Tuple[str, ...]
    dataset_kwargs: Dict[str, Any]
    target_attr: Optional[str] = None


@dataclass(frozen=True)
class MaskCBMRunPreset:
    dataset_key: str
    target_attr: Optional[str]

    n_groups: int
    include_strengths: bool

    # Mask-group learning hyperparameters
    mask_steps: int
    mask_lr: float
    mask_temp_start: float
    mask_temp_end: float
    mask_entropy_reg: float
    mask_smooth_reg: float
    mask_balance_reg: float
    mask_class_weight_balanced: bool

    # Sparse CBM head search preferences
    l1_ratios: Tuple[float, ...]
    target_active_concepts: int
    diversity_weight: float
    best_head_C: float
    best_head_l1_ratio: float

    # Optional label cleanup for unstable/rare classes.
    drop_rare_label_min_count: int = 0

    # Search behavior when preset is active.
    enable_concept_val_search: bool = False
    concept_search_skip_datasets: Tuple[str, ...] = ("office",)
    concept_search_max_candidates: int = 24

    # Optional faithfulness-focused controls.
    concept_max_corr: float = 0.995
    balance_concept_families: bool = False
    head_class_weight_balanced: bool = False
    faith_audit_max_k: int = 18
    faith_audit_max_nodes: int = 20

    # Optional objective shaping for concept-search on validation.
    faith_selection_weight: float = 0.0
    faith_selection_flip_weight: float = 0.0
    faith_selection_max_nodes: int = 12
    faith_selection_max_k: int = 12
    faith_selection_random_trials: int = 16


def resolve_gender_csv_path(root: Path, cwd: Optional[Path] = None) -> str:
    """Resolve genders.csv path using the same fallback order as the notebook."""
    base = Path.cwd() if cwd is None else Path(cwd)
    candidates = [
        base / "genders.csv",
        base / "data" / "genders.csv",
        base / "julia_code" / "genders.csv",
        root / "genders.csv",
        root / "data" / "genders.csv",
        root / "julia_code" / "genders.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(base / "julia_code" / "genders.csv")


def build_dataset_profiles(root: Path, cwd: Optional[Path] = None) -> Dict[str, DatasetProfile]:
    """Build dataset profiles, including a robust genders.csv fallback for SMS."""
    return {
        "highschool": DatasetProfile(
            candidates=("highschool", "sp_high_school"),
            dataset_kwargs={"network": "proximity"},
            target_attr="class",
        ),
        "office": DatasetProfile(
            candidates=("workplace", "sp_office"),
            dataset_kwargs={},
            target_attr="department",
        ),
        "hospital": DatasetProfile(
            candidates=("hospital", "sp_hospital"),
            dataset_kwargs={},
            target_attr="status",
        ),
        "temporal_clusters": DatasetProfile(
            candidates=("temporal_clusters",),
            dataset_kwargs={},
            target_attr=None,
        ),
        "sms": DatasetProfile(
            candidates=("copenhagen",),
            dataset_kwargs={
                "network": "sms",
                "time_attr": "timestamp",
                "gender_csv_path": resolve_gender_csv_path(root=root, cwd=cwd),
            },
            target_attr="female",
        ),
    }


BEST_RUN_PRESETS = {
    "office_best": MaskCBMRunPreset(
        dataset_key="office",
        target_attr="department",
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.004,
        mask_smooth_reg=0.03,
        mask_balance_reg=0.02,
        mask_class_weight_balanced=False,
        l1_ratios=(0.8, 1.0),
        target_active_concepts=8,
        diversity_weight=0.03,
        best_head_C=10.0,
        best_head_l1_ratio=0.8,
        drop_rare_label_min_count=0,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office",),
        concept_search_max_candidates=24,
    ),
    "highschool_class_best": MaskCBMRunPreset(
        dataset_key="highschool",
        target_attr="class",
        n_groups=12,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.008,
        mask_smooth_reg=0.03,
        mask_balance_reg=0.01,
        mask_class_weight_balanced=False,
        l1_ratios=(0.8, 1.0),
        target_active_concepts=8,
        diversity_weight=0.03,
        best_head_C=10.0,
        best_head_l1_ratio=0.8,
        drop_rare_label_min_count=0,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office", "highschool"),
        concept_search_max_candidates=24,
    ),
    "highschool_gender_best": MaskCBMRunPreset(
        dataset_key="highschool",
        target_attr="gender",
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.004,
        mask_smooth_reg=0.03,
        mask_balance_reg=0.02,
        mask_class_weight_balanced=False,
        l1_ratios=(1.0, 0.8),
        target_active_concepts=8,
        diversity_weight=0.03,
        best_head_C=0.31622776601683794,
        best_head_l1_ratio=0.8,
        drop_rare_label_min_count=10,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office", "highschool"),
        concept_search_max_candidates=24,
    ),
    "temporal_clusters_best": MaskCBMRunPreset(
        dataset_key="temporal_clusters",
        target_attr=None,
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.004,
        mask_smooth_reg=0.03,
        mask_balance_reg=0.02,
        mask_class_weight_balanced=False,
        l1_ratios=(0.8, 1.0),
        target_active_concepts=8,
        diversity_weight=0.03,
        best_head_C=10.0,
        best_head_l1_ratio=0.8,
        drop_rare_label_min_count=0,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office", "highschool"),
        concept_search_max_candidates=24,
    ),
    "hospital_best": MaskCBMRunPreset(
        dataset_key="hospital",
        target_attr="status",
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.008,
        mask_smooth_reg=0.015,
        mask_balance_reg=0.02,
        mask_class_weight_balanced=True,
        l1_ratios=(0.8, 1.0),
        target_active_concepts=8,
        diversity_weight=0.03,
        best_head_C=1.0,
        best_head_l1_ratio=1.0,
        drop_rare_label_min_count=0,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office", "highschool", "sms", "hospital"),
        concept_search_max_candidates=24,
        concept_max_corr=0.995,
        balance_concept_families=False,
        head_class_weight_balanced=False,
        faith_audit_max_k=18,
        faith_audit_max_nodes=20,
        faith_selection_weight=0.0,
        faith_selection_flip_weight=0.0,
        faith_selection_max_nodes=12,
        faith_selection_max_k=12,
        faith_selection_random_trials=16,
    ),
    "sms_best": MaskCBMRunPreset(
        dataset_key="sms",
        target_attr="female",
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.004,
        mask_smooth_reg=0.06,
        mask_balance_reg=0.02,
        mask_class_weight_balanced=False,
        l1_ratios=(0.8, 1.0),
        target_active_concepts=8,
        diversity_weight=0.03,
        best_head_C=0.31622776601683794,
        best_head_l1_ratio=0.8,
        drop_rare_label_min_count=0,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office", "highschool", "sms"),
        concept_search_max_candidates=24,
    ),
    "sms_faith_plus": MaskCBMRunPreset(
        dataset_key="sms",
        target_attr="female",
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.004,
        mask_smooth_reg=0.06,
        mask_balance_reg=0.02,
        mask_class_weight_balanced=True,
        l1_ratios=(0.8, 1.0),
        target_active_concepts=7,
        diversity_weight=0.05,
        best_head_C=0.31622776601683794,
        best_head_l1_ratio=0.8,
        drop_rare_label_min_count=0,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office", "highschool", "sms"),
        concept_search_max_candidates=24,
        concept_max_corr=0.985,
        balance_concept_families=False,
        head_class_weight_balanced=True,
        faith_audit_max_k=24,
        faith_audit_max_nodes=30,
    ),
    "sms_faith_sparse": MaskCBMRunPreset(
        dataset_key="sms",
        target_attr="female",
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.006,
        mask_smooth_reg=0.08,
        mask_balance_reg=0.03,
        mask_class_weight_balanced=True,
        l1_ratios=(1.0, 0.8),
        target_active_concepts=6,
        diversity_weight=0.06,
        best_head_C=0.1,
        best_head_l1_ratio=1.0,
        drop_rare_label_min_count=0,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office", "highschool", "sms"),
        concept_search_max_candidates=24,
        concept_max_corr=0.97,
        balance_concept_families=True,
        head_class_weight_balanced=True,
        faith_audit_max_k=28,
        faith_audit_max_nodes=40,
    ),
    "sms_faith_search": MaskCBMRunPreset(
        dataset_key="sms",
        target_attr="female",
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.004,
        mask_smooth_reg=0.06,
        mask_balance_reg=0.02,
        mask_class_weight_balanced=False,
        l1_ratios=(0.8, 1.0),
        target_active_concepts=8,
        diversity_weight=0.03,
        best_head_C=0.31622776601683794,
        best_head_l1_ratio=0.8,
        drop_rare_label_min_count=0,
        enable_concept_val_search=True,
        concept_search_skip_datasets=("office", "highschool"),
        concept_search_max_candidates=24,
        concept_max_corr=0.995,
        balance_concept_families=False,
        head_class_weight_balanced=False,
        faith_audit_max_k=18,
        faith_audit_max_nodes=20,
        faith_selection_weight=0.35,
        faith_selection_flip_weight=0.20,
        faith_selection_max_nodes=18,
        faith_selection_max_k=18,
        faith_selection_random_trials=16,
    ),
    "sms_compromise": MaskCBMRunPreset(
        dataset_key="sms",
        target_attr="female",
        n_groups=10,
        include_strengths=True,
        mask_steps=900,
        mask_lr=0.08,
        mask_temp_start=2.0,
        mask_temp_end=0.45,
        mask_entropy_reg=0.008,
        mask_smooth_reg=0.06,
        mask_balance_reg=0.01,
        mask_class_weight_balanced=False,
        l1_ratios=(0.8, 1.0),
        target_active_concepts=8,
        diversity_weight=0.03,
        best_head_C=0.31622776601683794,
        best_head_l1_ratio=0.8,
        drop_rare_label_min_count=0,
        enable_concept_val_search=False,
        concept_search_skip_datasets=("office", "highschool", "sms"),
        concept_search_max_candidates=24,
        concept_max_corr=0.995,
        balance_concept_families=False,
        head_class_weight_balanced=False,
        faith_audit_max_k=20,
        faith_audit_max_nodes=24,
        faith_selection_weight=0.0,
        faith_selection_flip_weight=0.0,
        faith_selection_max_nodes=12,
        faith_selection_max_k=12,
        faith_selection_random_trials=16,
    ),
}


DATASET_ALIASES = {
    "sp_high_school": "highschool",
    "sp_office": "office",
    "sp_hospital": "hospital",
    "temporal_clsuters": "temporal_clusters",  # common typo
}


def canonicalize_dataset_key(dataset_key: str) -> str:
    """Normalize user-facing dataset aliases to canonical keys."""
    return DATASET_ALIASES.get(dataset_key, dataset_key)


def apply_run_preset(
    dataset_key: str,
    target_attr_override: Optional[str],
    run_preset_key: Optional[str],
) -> Tuple[str, Optional[str], Optional[MaskCBMRunPreset]]:
    """Apply a named run preset and return updated (dataset_key, target_attr, preset)."""
    if run_preset_key is None:
        return dataset_key, target_attr_override, None

    if run_preset_key not in BEST_RUN_PRESETS:
        raise AssertionError(f"Unknown run_preset_key: {run_preset_key}")

    run_preset = BEST_RUN_PRESETS[run_preset_key]
    return run_preset.dataset_key, run_preset.target_attr, run_preset
