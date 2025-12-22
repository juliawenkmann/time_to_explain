#!/usr/bin/env python3
"""
Stick-figure contact-motion synthetic dataset for TGN (NON-bipartite) with ground-truth explanations.

Two motion types (balanced by default):
  type0: continuous clapping (wrists move together/apart)
  type1: legs splay apart and together

Contact rule:
  A contact edge appears every 4th frame when wrists (type0) or ankles (type1)
  are within a distance threshold.

TGN task framing:
  Predict the next interaction event: contact edges appear/disappear based on
  the deterministic distance rule above.

Outputs (default ./data):
  data/{name}.csv                 raw input to utils/preprocess_data.py
  data/{name}_gt_raw.json         ground truth mapping (raw row indices -> supporting row indices)
  data/{name}_gt.json             same mapping, but indices shifted by +1 (matches preprocess reindex convention)
  data/{name}_meta.json           metadata

Non-bipartite guarantee:
  Includes LS-RS edges, so triangle T-LS-RS exists.
"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import DatasetRecipe
from time_to_explain.core.registry import register_dataset
from time_to_explain.core.types import DatasetBundle


# Joint order per person
J_T, J_H, J_LS, J_RS, J_LE, J_RE, J_LW, J_RW, J_LH, J_RH, J_LK, J_RK, J_LA, J_RA = range(14)
JOINTS_PER_PERSON = 14

# Bone list (undirected conceptual), but we emit a single interaction per bone per frame
BONES = [
    (J_T,  J_H),   # torso-head
    (J_T,  J_LS),  # torso-left shoulder
    (J_T,  J_RS),  # torso-right shoulder
    (J_LS, J_RS),  # shoulders (makes triangle with torso -> non-bipartite)
    (J_LS, J_LE),  # left upper arm
    (J_LE, J_LW),  # left forearm
    (J_RS, J_RE),  # right upper arm
    (J_RE, J_RW),  # right forearm
    (J_T,  J_LH),  # torso-left hip
    (J_T,  J_RH),  # torso-right hip
    (J_LH, J_LK),  # left upper leg
    (J_LK, J_LA),  # left lower leg
    (J_RH, J_RK),  # right upper leg
    (J_RK, J_RA),  # right lower leg
]
ARM_SUPPORT_BONES = {
    (J_LS, J_LE),
    (J_LE, J_LW),
    (J_RS, J_RE),
    (J_RE, J_RW),
}
HIP_SUPPORT_BONES = {
    (J_RS, J_RE),
    (J_RE, J_RW),
}
LEG_SUPPORT_BONES = {
    (J_T, J_LH),
    (J_T, J_RH),
    (J_LH, J_LK),
    (J_LK, J_LA),
    (J_RH, J_RK),
    (J_RK, J_RA),
}

@dataclass
class Config:
    name: str = "stickwave"
    out_dir: str = "data"
    num_clips: int = 2000
    frames: int = 30
    dt: float = 1.0
    edge_feat_dim: int = 172
    node_feat_dim: int = 8
    node_feature_mode: str = "zeros"
    wave_freq: float = 1.0      # legacy (unused)
    wave_amp: float = 0.35      # motion span scalar
    jitter: float = 0.01        # coordinate noise
    contact_stride: int = 4
    contact_dist: float = 0.15
    seed: int = 0
    target_label: int = 1
    support_label: int = 0
    bipartite: bool = False
    explain_last_k_frames: int = 5  # ground truth: use last K frames of motion-related bones


def person_joint_id(person_id: int, joint: int) -> int:
    return person_id * JOINTS_PER_PERSON + joint


def kinematics_arm(shoulder_xy: np.ndarray, upper: float, fore: float, theta1: float, theta2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return elbow, wrist positions given shoulder position and angles."""
    elbow = shoulder_xy + upper * np.array([math.cos(theta1), math.sin(theta1)], dtype=np.float32)
    wrist = elbow + fore * np.array([math.cos(theta2), math.sin(theta2)], dtype=np.float32)
    return elbow, wrist


def two_link_ik(
    shoulder_xy: np.ndarray,
    target_xy: np.ndarray,
    upper: float,
    fore: float,
    *,
    elbow_up: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a planar 2-link IK for a target point."""
    dx = float(target_xy[0] - shoulder_xy[0])
    dy = float(target_xy[1] - shoulder_xy[1])
    r2 = dx * dx + dy * dy
    dist = math.sqrt(max(r2, 1e-8))
    max_reach = max(upper + fore - 1e-6, 1e-6)
    dist = min(dist, max_reach)

    cos_q2 = (dist * dist - upper * upper - fore * fore) / (2.0 * upper * fore)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    sin_q2 = math.sqrt(max(0.0, 1.0 - cos_q2 * cos_q2))
    if elbow_up:
        sin_q2 = -sin_q2
    q2 = math.atan2(sin_q2, cos_q2)

    q1 = math.atan2(dy, dx) - math.atan2(fore * sin_q2, upper + fore * cos_q2)
    elbow = shoulder_xy + upper * np.array([math.cos(q1), math.sin(q1)], dtype=np.float32)
    wrist = elbow + fore * np.array([math.cos(q1 + q2), math.sin(q1 + q2)], dtype=np.float32)
    return elbow, wrist


def build_clip_positions(cfg: Config, rng: np.random.Generator, clip_class: int) -> List[np.ndarray]:
    """
    Returns list of frames, each frame is (14,2) array of joint xy positions.
    clip_class:
      0 -> clapping (arms move)
      1 -> legs apart/together (legs move)
    """
    frames_xy: List[np.ndarray] = []

    # Base body in local coordinates
    torso = np.array([0.0, 0.0], dtype=np.float32)
    head  = np.array([0.0, 1.0], dtype=np.float32)
    LS    = np.array([-0.35, 0.80], dtype=np.float32)
    RS    = np.array([+0.35, 0.80], dtype=np.float32)
    LH    = np.array([-0.20, -0.35], dtype=np.float32)
    RH    = np.array([+0.20, -0.35], dtype=np.float32)

    upper = 0.35
    fore  = 0.35
    upper_leg = 0.45
    lower_leg = 0.45

    # Fixed limbs (used for the non-moving parts).
    LE_fixed, LW_fixed = kinematics_arm(LS, upper, fore, -math.pi/2, -math.pi/2)
    RE_fixed, RW_fixed = kinematics_arm(RS, upper, fore, -math.pi/2, -math.pi/2)
    theta_leg_l = -math.pi/2 - 0.05
    theta_leg_r = -math.pi/2 + 0.05
    LK_fixed, LA_fixed = kinematics_arm(LH, upper_leg, lower_leg, theta_leg_l, theta_leg_l)
    RK_fixed, RA_fixed = kinematics_arm(RH, upper_leg, lower_leg, theta_leg_r, theta_leg_r)

    clap_open = 0.35 + float(cfg.wave_amp)
    clap_closed = 0.05
    clap_y = 0.6

    leg_open = 0.20 + float(cfg.wave_amp)
    leg_closed = 0.05
    leg_y = -1.1

    contact_stride = max(1, int(cfg.contact_stride))

    # Randomize global transform per clip (prevents trivial memorization)
    scale = float(rng.uniform(0.8, 1.2))
    shift = rng.normal(loc=0.0, scale=0.2, size=(2,)).astype(np.float32)
    # Small random “camera rotation” in 2D
    rot = float(rng.uniform(-0.2, 0.2))
    R = np.array([[math.cos(rot), -math.sin(rot)],
                  [math.sin(rot),  math.cos(rot)]], dtype=np.float32)

    jitter = rng.normal(0.0, cfg.jitter, size=(JOINTS_PER_PERSON, 2)).astype(np.float32) if cfg.jitter > 0 else 0.0

    for f in range(cfg.frames):
        closeness = 0.5 * (1.0 + math.cos(2 * math.pi * (f / contact_stride)))

        if clip_class == 0:
            left_x = -(clap_open * (1.0 - closeness) + clap_closed * closeness)
            right_x = +(clap_open * (1.0 - closeness) + clap_closed * closeness)
            left_target = np.array([left_x, clap_y], dtype=np.float32)
            right_target = np.array([right_x, clap_y], dtype=np.float32)
            LE, LW = two_link_ik(LS, left_target, upper, fore, elbow_up=False)
            RE, RW = two_link_ik(RS, right_target, upper, fore, elbow_up=True)
            LK, LA = LK_fixed, LA_fixed
            RK, RA = RK_fixed, RA_fixed
        else:
            LE, LW = LE_fixed, LW_fixed
            RE, RW = RE_fixed, RW_fixed
            left_x = -(leg_open * (1.0 - closeness) + leg_closed * closeness)
            right_x = +(leg_open * (1.0 - closeness) + leg_closed * closeness)
            left_target = np.array([left_x, leg_y], dtype=np.float32)
            right_target = np.array([right_x, leg_y], dtype=np.float32)
            LK, LA = two_link_ik(LH, left_target, upper_leg, lower_leg, elbow_up=False)
            RK, RA = two_link_ik(RH, right_target, upper_leg, lower_leg, elbow_up=True)

        xy = np.zeros((JOINTS_PER_PERSON, 2), dtype=np.float32)
        xy[J_T]  = torso
        xy[J_H]  = head
        xy[J_LS] = LS
        xy[J_RS] = RS
        xy[J_LE] = LE
        xy[J_LW] = LW
        xy[J_RE] = RE
        xy[J_RW] = RW
        xy[J_LH] = LH
        xy[J_RH] = RH
        xy[J_LK] = LK
        xy[J_LA] = LA
        xy[J_RK] = RK
        xy[J_RA] = RA

        # Apply global transform + jitter
        xy = (xy @ R.T) * scale + shift
        if isinstance(jitter, np.ndarray):
            xy += jitter

        frames_xy.append(xy)

    return frames_xy


def build_clip_positions_hips(cfg: Config, rng: np.random.Generator) -> List[np.ndarray]:
    """
    Returns list of frames for sticky_hips, each frame is (14,2) array of joint xy positions.
    Right arm moves toward the hip/torso side; left arm stays fixed.
    """
    frames_xy: List[np.ndarray] = []

    # Base body in local coordinates
    torso = np.array([0.0, 0.0], dtype=np.float32)
    head  = np.array([0.0, 1.0], dtype=np.float32)
    LS    = np.array([-0.35, 0.80], dtype=np.float32)
    RS    = np.array([+0.35, 0.80], dtype=np.float32)
    LH    = np.array([-0.20, -0.35], dtype=np.float32)
    RH    = np.array([+0.20, -0.35], dtype=np.float32)

    upper = 0.35
    fore  = 0.35
    upper_leg = 0.45
    lower_leg = 0.45

    # Fixed limbs for the non-moving parts.
    LE_fixed, LW_fixed = kinematics_arm(LS, upper, fore, -math.pi / 2, -math.pi / 2)
    theta_leg_l = -math.pi / 2 - 0.05
    theta_leg_r = -math.pi / 2 + 0.05
    LK_fixed, LA_fixed = kinematics_arm(LH, upper_leg, lower_leg, theta_leg_l, theta_leg_l)
    RK_fixed, RA_fixed = kinematics_arm(RH, upper_leg, lower_leg, theta_leg_r, theta_leg_r)

    contact_stride = max(1, int(cfg.contact_stride))
    open_target = np.array([0.85 + 0.2 * float(cfg.wave_amp), 0.85], dtype=np.float32)
    hip_target = np.array([0.25, 0.15], dtype=np.float32)

    # Randomize global transform per clip (prevents trivial memorization)
    scale = float(rng.uniform(0.8, 1.2))
    shift = rng.normal(loc=0.0, scale=0.2, size=(2,)).astype(np.float32)
    rot = float(rng.uniform(-0.2, 0.2))
    R = np.array(
        [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]],
        dtype=np.float32,
    )

    jitter = (
        rng.normal(0.0, cfg.jitter, size=(JOINTS_PER_PERSON, 2)).astype(np.float32)
        if cfg.jitter > 0
        else 0.0
    )

    for f in range(cfg.frames):
        closeness = 0.5 * (1.0 + math.cos(2 * math.pi * (f / contact_stride)))
        target = open_target * (1.0 - closeness) + hip_target * closeness

        RE, RW = two_link_ik(RS, target, upper, fore, elbow_up=True)
        LE, LW = LE_fixed, LW_fixed
        LK, LA = LK_fixed, LA_fixed
        RK, RA = RK_fixed, RA_fixed

        xy = np.zeros((JOINTS_PER_PERSON, 2), dtype=np.float32)
        xy[J_T]  = torso
        xy[J_H]  = head
        xy[J_LS] = LS
        xy[J_RS] = RS
        xy[J_LE] = LE
        xy[J_LW] = LW
        xy[J_RE] = RE
        xy[J_RW] = RW
        xy[J_LH] = LH
        xy[J_RH] = RH
        xy[J_LK] = LK
        xy[J_LA] = LA
        xy[J_RK] = RK
        xy[J_RA] = RA

        # Apply global transform + jitter
        xy = (xy @ R.T) * scale + shift
        if isinstance(jitter, np.ndarray):
            xy += jitter

        frames_xy.append(xy)

    return frames_xy


def make_edge_feat(
    cfg: Config,
    xu: float,
    yu: float,
    xv: float,
    yv: float,
    bone_id: int,
    is_query: int,
    frame_norm: float,
) -> List[float]:
    """
    172-dim edge feature vector (first few dims used, rest zeros).
    """
    dx = xv - xu
    dy = yv - yu
    feat = [0.0] * cfg.edge_feat_dim
    values = [
        float(xu),
        float(yu),
        float(xv),
        float(yv),
        float(dx),
        float(dy),
        float(bone_id) / 16.0,
        float(is_query),
        float(frame_norm),
    ]
    for idx, value in enumerate(values):
        if idx >= cfg.edge_feat_dim:
            break
        feat[idx] = value
    return feat


def build_stick_figure_explain_index(
    cfg: Dict[str, object] | Config,
    *,
    last_k_frames: Optional[int] = None,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Build explain indices + ground-truth supports using the stick-figure event layout.

    Returns 0-based row indices in the generated interaction order.
    """
    if isinstance(cfg, Config):
        cfg_dict = asdict(cfg)
    else:
        cfg_dict = dict(cfg)

    if last_k_frames is not None:
        cfg_dict["explain_last_k_frames"] = int(last_k_frames)

    gen_cfg = Config(
        num_clips=int(cfg_dict.get("num_clips", 0)),
        frames=int(cfg_dict.get("frames", 0)),
        dt=float(cfg_dict.get("dt", 1.0)),
        edge_feat_dim=int(cfg_dict.get("edge_feat_dim", 0)),
        node_feat_dim=int(cfg_dict.get("node_feat_dim", 0)),
        node_feature_mode=str(cfg_dict.get("node_feature_mode", "zeros")),
        wave_freq=float(cfg_dict.get("wave_freq", 1.0)),
        wave_amp=float(cfg_dict.get("wave_amp", 0.35)),
        jitter=float(cfg_dict.get("jitter", 0.0)),
        contact_stride=int(cfg_dict.get("contact_stride", 4)),
        contact_dist=float(cfg_dict.get("contact_dist", 0.15)),
        seed=int(cfg_dict.get("seed", 0)),
        target_label=int(cfg_dict.get("target_label", 1)),
        support_label=int(cfg_dict.get("support_label", 0)),
        explain_last_k_frames=int(cfg_dict.get("explain_last_k_frames", 1)),
        bipartite=bool(cfg_dict.get("bipartite", False)),
    )

    if gen_cfg.num_clips <= 0 or gen_cfg.frames <= 0:
        raise ValueError("Config must include positive num_clips and frames values.")

    rng = np.random.default_rng(gen_cfg.seed)
    _rows, _edge_feats, gt_raw, _target_classes = build_stick_figure_events(gen_cfg, rng)
    explain_idxs = sorted(int(k) for k in gt_raw.keys())
    return explain_idxs, gt_raw


def build_sticky_hips_explain_index(
    cfg: Dict[str, object] | Config,
    *,
    last_k_frames: Optional[int] = None,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Build explain indices + ground-truth supports for sticky_hips.

    Returns 0-based row indices in the generated interaction order.
    """
    if isinstance(cfg, Config):
        cfg_dict = asdict(cfg)
    else:
        cfg_dict = dict(cfg)

    if last_k_frames is not None:
        cfg_dict["explain_last_k_frames"] = int(last_k_frames)

    gen_cfg = Config(
        num_clips=int(cfg_dict.get("num_clips", 0)),
        frames=int(cfg_dict.get("frames", 0)),
        dt=float(cfg_dict.get("dt", 1.0)),
        edge_feat_dim=int(cfg_dict.get("edge_feat_dim", 0)),
        node_feat_dim=int(cfg_dict.get("node_feat_dim", 0)),
        node_feature_mode=str(cfg_dict.get("node_feature_mode", "zeros")),
        wave_freq=float(cfg_dict.get("wave_freq", 1.0)),
        wave_amp=float(cfg_dict.get("wave_amp", 0.35)),
        jitter=float(cfg_dict.get("jitter", 0.0)),
        contact_stride=int(cfg_dict.get("contact_stride", 4)),
        contact_dist=float(cfg_dict.get("contact_dist", 0.4)),
        seed=int(cfg_dict.get("seed", 0)),
        target_label=int(cfg_dict.get("target_label", 1)),
        support_label=int(cfg_dict.get("support_label", 0)),
        explain_last_k_frames=int(cfg_dict.get("explain_last_k_frames", 1)),
        bipartite=bool(cfg_dict.get("bipartite", False)),
    )

    if gen_cfg.num_clips <= 0 or gen_cfg.frames <= 0:
        raise ValueError("Config must include positive num_clips and frames values.")

    rng = np.random.default_rng(gen_cfg.seed)
    _rows, _edge_feats, gt_raw, _target_classes = build_sticky_hips_events(gen_cfg, rng)
    explain_idxs = sorted(int(k) for k in gt_raw.keys())
    return explain_idxs, gt_raw


def build_stick_figure_events(
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[List[List[float]], List[List[float]], Dict[int, List[int]], Dict[int, int]]:
    # Balanced random motion types
    classes = np.array(
        [0] * (cfg.num_clips // 2) + [1] * (cfg.num_clips - cfg.num_clips // 2),
        dtype=np.int32,
    )
    rng.shuffle(classes)

    rows: List[List[float]] = []
    edge_features: List[List[float]] = []
    gt_raw: Dict[int, List[int]] = {}
    target_classes: Dict[int, int] = {}

    t = 0.0

    # We generate clips sequentially; class is randomized so no time/class correlation.
    for clip_id in range(cfg.num_clips):
        c = int(classes[clip_id])

        frames_xy = build_clip_positions(cfg, rng, c)

        support_bones = ARM_SUPPORT_BONES if c == 0 else LEG_SUPPORT_BONES
        contact_stride = max(1, int(cfg.contact_stride))
        contact_dist = float(cfg.contact_dist)
        if c == 0:
            contact_joint_a, contact_joint_b = J_LW, J_RW
            contact_bone_id = 14
        else:
            contact_joint_a, contact_joint_b = J_LA, J_RA
            contact_bone_id = 15

        # Track indices of motion-related edges for GT.
        per_frame_support_indices: List[List[int]] = []
        contact_events: List[Tuple[int, int]] = []

        for f in range(cfg.frames):
            xy = frames_xy[f]
            frame_norm = f / max(1, cfg.frames - 1)

            support_this_frame: List[int] = []

            for bone_id, (a, b) in enumerate(BONES):
                u = person_joint_id(clip_id, a)
                v = person_joint_id(clip_id, b)

                xu, yu = float(xy[a, 0]), float(xy[a, 1])
                xv, yv = float(xy[b, 0]), float(xy[b, 1])

                feat = make_edge_feat(cfg, xu, yu, xv, yv, bone_id=bone_id, is_query=0, frame_norm=frame_norm)

                idx = len(rows)
                rows.append([u, v, t, float(cfg.support_label)])
                edge_features.append(feat)
                t += cfg.dt

                # If it's one of the support bones, record it
                if (a, b) in support_bones:
                    support_this_frame.append(idx)

            per_frame_support_indices.append(support_this_frame)

            # Contact edge: appears every contact_stride frames when limbs are close.
            if f % contact_stride == 0:
                xa, ya = float(xy[contact_joint_a, 0]), float(xy[contact_joint_a, 1])
                xb, yb = float(xy[contact_joint_b, 0]), float(xy[contact_joint_b, 1])
                dist = float(np.hypot(xa - xb, ya - yb))
                if dist <= contact_dist:
                    u = person_joint_id(clip_id, contact_joint_a)
                    v = person_joint_id(clip_id, contact_joint_b)
                    feat = make_edge_feat(
                        cfg,
                        xa,
                        ya,
                        xb,
                        yb,
                        bone_id=contact_bone_id,
                        is_query=1,
                        frame_norm=frame_norm,
                    )
                    query_idx = len(rows)
                    rows.append([u, v, t, float(cfg.target_label)])
                    edge_features.append(feat)
                    t += cfg.dt
                    contact_events.append((query_idx, f))

        # Ground-truth explanation: last K frames of the motion-related bones
        k_frames = min(max(1, cfg.explain_last_k_frames), cfg.frames)
        for query_idx, f in contact_events:
            start = max(0, f - k_frames + 1)
            support_idxs: List[int] = []
            for f_idx in range(start, f + 1):
                support_idxs.extend(per_frame_support_indices[f_idx])
            gt_raw[query_idx] = support_idxs
            target_classes[query_idx] = c

    return rows, edge_features, gt_raw, target_classes


def build_sticky_hips_events(
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[List[List[float]], List[List[float]], Dict[int, List[int]], Dict[int, int]]:
    rows: List[List[float]] = []
    edge_features: List[List[float]] = []
    gt_raw: Dict[int, List[int]] = {}
    target_classes: Dict[int, int] = {}

    t = 0.0
    contact_stride = max(1, int(cfg.contact_stride))
    contact_dist = float(cfg.contact_dist)
    contact_joint_a, contact_joint_b = J_RW, J_T
    contact_bone_id = 16

    for clip_id in range(cfg.num_clips):
        frames_xy = build_clip_positions_hips(cfg, rng)
        support_bones = HIP_SUPPORT_BONES

        per_frame_support_indices: List[List[int]] = []
        contact_events: List[Tuple[int, int]] = []

        for f in range(cfg.frames):
            xy = frames_xy[f]
            frame_norm = f / max(1, cfg.frames - 1)
            support_this_frame: List[int] = []

            for bone_id, (a, b) in enumerate(BONES):
                u = person_joint_id(clip_id, a)
                v = person_joint_id(clip_id, b)

                xu, yu = float(xy[a, 0]), float(xy[a, 1])
                xv, yv = float(xy[b, 0]), float(xy[b, 1])

                feat = make_edge_feat(
                    cfg,
                    xu,
                    yu,
                    xv,
                    yv,
                    bone_id=bone_id,
                    is_query=0,
                    frame_norm=frame_norm,
                )

                idx = len(rows)
                rows.append([u, v, t, float(cfg.support_label)])
                edge_features.append(feat)
                t += cfg.dt

                if (a, b) in support_bones:
                    support_this_frame.append(idx)

            per_frame_support_indices.append(support_this_frame)

            if f % contact_stride == 0:
                xa, ya = float(xy[contact_joint_a, 0]), float(xy[contact_joint_a, 1])
                xb, yb = float(xy[contact_joint_b, 0]), float(xy[contact_joint_b, 1])
                dist = float(np.hypot(xa - xb, ya - yb))
                if dist <= contact_dist:
                    u = person_joint_id(clip_id, contact_joint_a)
                    v = person_joint_id(clip_id, contact_joint_b)
                    feat = make_edge_feat(
                        cfg,
                        xa,
                        ya,
                        xb,
                        yb,
                        bone_id=contact_bone_id,
                        is_query=1,
                        frame_norm=frame_norm,
                    )
                    query_idx = len(rows)
                    rows.append([u, v, t, float(cfg.target_label)])
                    edge_features.append(feat)
                    t += cfg.dt
                    contact_events.append((query_idx, f))

        k_frames = min(max(1, cfg.explain_last_k_frames), cfg.frames)
        for query_idx, f in contact_events:
            start = max(0, f - k_frames + 1)
            support_idxs: List[int] = []
            for f_idx in range(start, f + 1):
                support_idxs.extend(per_frame_support_indices[f_idx])
            gt_raw[query_idx] = support_idxs
            target_classes[query_idx] = 0

    return rows, edge_features, gt_raw, target_classes


@register_dataset("stick_figure")
class StickFigure(DatasetRecipe):
    @classmethod
    def default_config(cls) -> Dict[str, object]:
        return dict(
            num_clips=2000,
            frames=30,
            dt=1.0,
            edge_feat_dim=9,
            node_feat_dim=8,
            node_feature_mode="zeros",
            wave_freq=1.0,
            wave_amp=0.35,
            jitter=0.01,
            contact_stride=4,
            contact_dist=0.15,
            seed=0,
            target_label=1,
            support_label=0,
            explain_last_k_frames=5,
            bipartite=False,
        )

    def generate(self) -> DatasetBundle:
        cfg = {**self.default_config(), **(self.config or {})}

        gen_cfg = Config(
            num_clips=int(cfg["num_clips"]),
            frames=int(cfg["frames"]),
            dt=float(cfg["dt"]),
            edge_feat_dim=int(cfg["edge_feat_dim"]),
            node_feat_dim=int(cfg["node_feat_dim"]),
            node_feature_mode=str(cfg.get("node_feature_mode", "zeros")),
            wave_freq=float(cfg["wave_freq"]),
            wave_amp=float(cfg["wave_amp"]),
            jitter=float(cfg["jitter"]),
            contact_stride=int(cfg.get("contact_stride", 4)),
            contact_dist=float(cfg.get("contact_dist", 0.15)),
            seed=int(cfg["seed"]),
            target_label=int(cfg.get("target_label", 1)),
            support_label=int(cfg.get("support_label", 0)),
            explain_last_k_frames=int(cfg.get("explain_last_k_frames", 5)),
            bipartite=bool(cfg.get("bipartite", False)),
        )

        rng = np.random.default_rng(gen_cfg.seed)
        rows, edge_feats_list, gt_raw, target_classes = build_stick_figure_events(gen_cfg, rng)

        df = pd.DataFrame(rows, columns=["u", "i", "ts", "label"])
        df["event_id"] = np.arange(len(df), dtype=int)
        df = df.sort_values("ts").reset_index(drop=True)
        order = df["event_id"].to_numpy()

        if gen_cfg.edge_feat_dim > 0 and edge_feats_list:
            edge_features = np.asarray(edge_feats_list, dtype=float)
            edge_features = edge_features[order]
        else:
            edge_features = None

        old_to_new = {int(old): int(new) for new, old in enumerate(order.tolist())}
        targets = [old_to_new[int(k)] for k in gt_raw.keys()]
        rationales = {
            old_to_new[int(k)]: [old_to_new[int(v)] for v in vs] for k, vs in gt_raw.items()
        }
        target_classes = {old_to_new[int(k)]: int(v) for k, v in target_classes.items()}

        df = df.drop(columns=["event_id"])

        num_nodes = int(gen_cfg.num_clips) * JOINTS_PER_PERSON
        if gen_cfg.node_feature_mode.lower() in {"zeros", "zero"}:
            node_features = np.zeros((num_nodes, int(gen_cfg.node_feat_dim)), dtype=float)
        elif gen_cfg.node_feature_mode.lower() in {"random", "rand"}:
            node_features = rng.normal(size=(num_nodes, int(gen_cfg.node_feat_dim)))
        else:
            raise ValueError(f"Unknown node_feature_mode '{gen_cfg.node_feature_mode}'.")

        meta: Dict[str, object] = {
            "recipe": "stick_figure",
            "config": cfg,
            "description": "Stick-figure with static body. Contact edge appears every contact_stride frames when wrists (clap) or ankles (leg splay) are close.",
            "class_rule": {
                "0": "clapping => contact edge between left/right wrists",
                "1": "leg splay => contact edge between left/right ankles",
            },
            "non_bipartite_note": "Triangle exists via edges (T,LS), (T,RS), (LS,RS).",
            "num_rows": int(len(df)),
            "num_queries": int(len(gt_raw)),
            "node_feat_dim": int(gen_cfg.node_feat_dim),
            "edge_feat_dim": int(gen_cfg.edge_feat_dim),
            "bipartite": bool(gen_cfg.bipartite),
            "ground_truth": {
                "targets": targets,
                "rationales": rationales,
                "target_classes": target_classes,
                "label_semantics": "label==target_label marks the contact edges (targets).",
                "explanation_semantics": "Targets are explained by motion-related bones in the last K frames.",
            },
        }

        return {
            "interactions": df,
            "node_features": node_features,
            "edge_features": edge_features,
            "metadata": meta,
        }


@register_dataset("sticky_hips")
class StickyHips(DatasetRecipe):
    @classmethod
    def default_config(cls) -> Dict[str, object]:
        return dict(
            num_clips=2000,
            frames=30,
            dt=1.0,
            edge_feat_dim=9,
            node_feat_dim=8,
            node_feature_mode="zeros",
            wave_freq=1.0,
            wave_amp=0.35,
            jitter=0.01,
            contact_stride=4,
            contact_dist=0.4,
            seed=0,
            target_label=1,
            support_label=0,
            explain_last_k_frames=5,
            bipartite=False,
        )

    def generate(self) -> DatasetBundle:
        cfg = {**self.default_config(), **(self.config or {})}

        gen_cfg = Config(
            num_clips=int(cfg["num_clips"]),
            frames=int(cfg["frames"]),
            dt=float(cfg["dt"]),
            edge_feat_dim=int(cfg["edge_feat_dim"]),
            node_feat_dim=int(cfg["node_feat_dim"]),
            node_feature_mode=str(cfg.get("node_feature_mode", "zeros")),
            wave_freq=float(cfg["wave_freq"]),
            wave_amp=float(cfg["wave_amp"]),
            jitter=float(cfg["jitter"]),
            contact_stride=int(cfg.get("contact_stride", 4)),
            contact_dist=float(cfg.get("contact_dist", 0.4)),
            seed=int(cfg["seed"]),
            target_label=int(cfg.get("target_label", 1)),
            support_label=int(cfg.get("support_label", 0)),
            explain_last_k_frames=int(cfg.get("explain_last_k_frames", 5)),
            bipartite=bool(cfg.get("bipartite", False)),
        )

        rng = np.random.default_rng(gen_cfg.seed)
        rows, edge_feats_list, gt_raw, target_classes = build_sticky_hips_events(gen_cfg, rng)

        df = pd.DataFrame(rows, columns=["u", "i", "ts", "label"])
        df["event_id"] = np.arange(len(df), dtype=int)
        df = df.sort_values("ts").reset_index(drop=True)
        order = df["event_id"].to_numpy()

        if gen_cfg.edge_feat_dim > 0 and edge_feats_list:
            edge_features = np.asarray(edge_feats_list, dtype=float)
            edge_features = edge_features[order]
        else:
            edge_features = None

        old_to_new = {int(old): int(new) for new, old in enumerate(order.tolist())}
        targets = [old_to_new[int(k)] for k in gt_raw.keys()]
        rationales = {
            old_to_new[int(k)]: [old_to_new[int(v)] for v in vs] for k, vs in gt_raw.items()
        }
        target_classes = {old_to_new[int(k)]: int(v) for k, v in target_classes.items()}

        df = df.drop(columns=["event_id"])

        num_nodes = int(gen_cfg.num_clips) * JOINTS_PER_PERSON
        if gen_cfg.node_feature_mode.lower() in {"zeros", "zero"}:
            node_features = np.zeros((num_nodes, int(gen_cfg.node_feat_dim)), dtype=float)
        elif gen_cfg.node_feature_mode.lower() in {"random", "rand"}:
            node_features = rng.normal(size=(num_nodes, int(gen_cfg.node_feat_dim)))
        else:
            raise ValueError(f"Unknown node_feature_mode '{gen_cfg.node_feature_mode}'.")

        meta: Dict[str, object] = {
            "recipe": "sticky_hips",
            "config": cfg,
            "description": (
                "Stick-figure with static body. Right arm moves toward the hip/torso side; "
                "a contact edge appears between right wrist and torso every contact_stride frames "
                "when they are close."
            ),
            "class_rule": {"0": "right-arm-to-hip => contact edge between right wrist and torso"},
            "non_bipartite_note": "Triangle exists via edges (T,LS), (T,RS), (LS,RS).",
            "num_rows": int(len(df)),
            "num_queries": int(len(gt_raw)),
            "node_feat_dim": int(gen_cfg.node_feat_dim),
            "edge_feat_dim": int(gen_cfg.edge_feat_dim),
            "bipartite": bool(gen_cfg.bipartite),
            "ground_truth": {
                "targets": targets,
                "rationales": rationales,
                "target_classes": target_classes,
                "label_semantics": "label==target_label marks the contact edges (targets).",
                "explanation_semantics": "Targets are explained by right-arm bones in the last K frames.",
            },
        }

        return {
            "interactions": df,
            "node_features": node_features,
            "edge_features": edge_features,
            "metadata": meta,
        }


def write_csv(
    path: Path,
    rows: List[List[float]],
    edge_features: List[List[float]],
    edge_feat_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["u", "i", "ts", "label"] + [f"f{k}" for k in range(edge_feat_dim)]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for idx, r in enumerate(rows):
            if edge_feat_dim > 0:
                feat = edge_features[idx]
                f.write(",".join(str(x) for x in (r + feat)) + "\n")
            else:
                f.write(",".join(str(x) for x in r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="stickwave")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--num_clips", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--edge_feat_dim", type=int, default=172)
    ap.add_argument("--node_feat_dim", type=int, default=8)
    ap.add_argument("--node_feature_mode", type=str, default="zeros")
    ap.add_argument("--wave_freq", type=float, default=1.0)
    ap.add_argument("--wave_amp", type=float, default=0.35)
    ap.add_argument("--jitter", type=float, default=0.01)
    ap.add_argument("--contact_stride", type=int, default=4)
    ap.add_argument("--contact_dist", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_label", type=int, default=1)
    ap.add_argument("--support_label", type=int, default=0)
    ap.add_argument("--explain_last_k_frames", type=int, default=5)
    args = ap.parse_args()

    cfg = Config(
        name=args.name,
        out_dir=args.out_dir,
        num_clips=args.num_clips,
        frames=args.frames,
        dt=args.dt,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=args.node_feat_dim,
        node_feature_mode=str(args.node_feature_mode),
        wave_freq=args.wave_freq,
        wave_amp=args.wave_amp,
        jitter=args.jitter,
        contact_stride=args.contact_stride,
        contact_dist=args.contact_dist,
        seed=args.seed,
        target_label=args.target_label,
        support_label=args.support_label,
        explain_last_k_frames=args.explain_last_k_frames,
    )

    rng = np.random.default_rng(cfg.seed)
    rows, edge_features, gt_raw, _target_classes = build_stick_figure_events(cfg, rng)

    out = Path(cfg.out_dir)
    csv_path = out / f"{cfg.name}.csv"
    write_csv(csv_path, rows, edge_features, cfg.edge_feat_dim)

    # Many TGN preprocess scripts reindex event idx/node ids by +1 (reserve 0). Save both forms.
    gt_processed = {str(int(k) + 1): [int(x) + 1 for x in v] for k, v in gt_raw.items()}

    (out / f"{cfg.name}_gt_raw.json").write_text(json.dumps(gt_raw, indent=2), encoding="utf-8")
    (out / f"{cfg.name}_gt.json").write_text(json.dumps(gt_processed, indent=2), encoding="utf-8")

    meta = {
        "config": asdict(cfg),
        "description": "Stick-figure with static body. Contact edge appears every contact_stride frames when wrists (clap) or ankles (leg splay) are close.",
        "class_rule": {"0": "clapping => contact edge between left/right wrists",
                       "1": "leg splay => contact edge between left/right ankles"},
        "non_bipartite_note": "Triangle exists via edges (T,LS), (T,RS), (LS,RS).",
        "num_rows": len(rows),
        "num_queries": len(gt_raw),
    }
    (out / f"{cfg.name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(" ", csv_path)
    print(" ", out / f"{cfg.name}_gt_raw.json")
    print(" ", out / f"{cfg.name}_gt.json")
    print(" ", out / f"{cfg.name}_meta.json")
    print(f"Total events: {len(rows):,} | Queries: {len(gt_raw):,}")


def write_stick_figure_explain_index(
    ml_csv: Path,
    out_csv: Path,
    *,
    ground_truth: Optional[Dict[str, object]] = None,
    gt_json: Optional[Path] = None,
    config: Optional[Dict[str, object] | Config] = None,
    explain_idxs: Optional[List[int]] = None,
    gt_raw: Optional[Dict[int, List[int]]] = None,
    last_k_frames: Optional[int] = None,
    test_split: float = 0.85,
    overwrite: bool = True,
    gt_raw_out: Optional[Path] = None,
    gt_out: Optional[Path] = None,
) -> Path:
    """
    Create an explain-index CSV for stick_figure from ground-truth query indices.

    If ground_truth is provided, uses its 0-based targets and rationales.
    If config is provided, rebuilds supports deterministically via generation.
    If gt_json is provided, assumes its keys are already 1-based.
    """
    out_csv = Path(out_csv)
    if out_csv.exists() and not overwrite:
        return out_csv

    if gt_raw is None:
        if ground_truth is not None:
            raw_targets = [int(t) for t in (ground_truth.get("targets") or [])]
            raw_rationales = ground_truth.get("rationales") or {}
            gt_raw = {}
            for target in raw_targets:
                support = raw_rationales.get(str(target), raw_rationales.get(target, []))
                support_list = [int(x) for x in (support or [])]
                gt_raw[int(target)] = support_list
        elif config is not None:
            explain_idxs, gt_raw = build_stick_figure_explain_index(
                config,
                last_k_frames=last_k_frames,
            )
        elif gt_json is not None:
            gt_data = json.loads(Path(gt_json).read_text(encoding="utf-8"))
            gt_raw = {
                int(k) - 1: [int(x) - 1 for x in v]
                for k, v in gt_data.items()
            }
        else:
            raise ValueError("Provide ground_truth, config, gt_raw, or gt_json.")

    if explain_idxs is None:
        explain_idxs = sorted(int(k) for k in gt_raw.keys())

    if gt_raw_out is not None:
        gt_raw_out = Path(gt_raw_out)
        gt_raw_out.parent.mkdir(parents=True, exist_ok=True)
        gt_raw_serial = {str(int(k)): [int(x) for x in v] for k, v in gt_raw.items()}
        gt_raw_out.write_text(json.dumps(gt_raw_serial, indent=2), encoding="utf-8")

    if gt_out is not None:
        gt_out = Path(gt_out)
        gt_out.parent.mkdir(parents=True, exist_ok=True)
        gt_proc = {str(int(k) + 1): [int(x) + 1 for x in v] for k, v in gt_raw.items()}
        gt_out.write_text(json.dumps(gt_proc, indent=2), encoding="utf-8")

    df = pd.read_csv(ml_csv)
    idx_col = "idx" if "idx" in df.columns else ("e_idx" if "e_idx" in df.columns else None)
    if idx_col is None:
        raise ValueError("Processed CSV is missing 'idx'/'e_idx' column required for explain indices.")

    n = len(df)
    test_start = int(float(test_split) * n)
    test_idx_set = set(df.loc[test_start:, idx_col].astype(int).tolist())
    query_idxs = [int(k) + 1 for k in explain_idxs]
    query_idxs_test = [k for k in query_idxs if k in test_idx_set] or query_idxs

    cols = [idx_col, "u", "i", "ts"]
    if "label" in df.columns:
        cols.append("label")
    explain_df = df[df[idx_col].isin(query_idxs_test)][cols].sort_values(idx_col)
    explain_df = explain_df.rename(columns={idx_col: "idx"})
    explain_df.insert(0, "event_idx", explain_df["idx"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    explain_df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv, " (#items:", len(explain_df), ")")
    return out_csv


def write_sticky_hips_explain_index(
    ml_csv: Path,
    out_csv: Path,
    *,
    ground_truth: Optional[Dict[str, object]] = None,
    gt_json: Optional[Path] = None,
    config: Optional[Dict[str, object] | Config] = None,
    explain_idxs: Optional[List[int]] = None,
    gt_raw: Optional[Dict[int, List[int]]] = None,
    last_k_frames: Optional[int] = None,
    test_split: float = 0.85,
    overwrite: bool = True,
    gt_raw_out: Optional[Path] = None,
    gt_out: Optional[Path] = None,
) -> Path:
    """
    Create an explain-index CSV for sticky_hips from ground-truth query indices.
    """
    out_csv = Path(out_csv)
    if out_csv.exists() and not overwrite:
        return out_csv

    if gt_raw is None:
        if ground_truth is not None:
            raw_targets = [int(t) for t in (ground_truth.get("targets") or [])]
            raw_rationales = ground_truth.get("rationales") or {}
            gt_raw = {}
            for target in raw_targets:
                support = raw_rationales.get(str(target), raw_rationales.get(target, []))
                support_list = [int(x) for x in (support or [])]
                gt_raw[int(target)] = support_list
        elif config is not None:
            explain_idxs, gt_raw = build_sticky_hips_explain_index(
                config,
                last_k_frames=last_k_frames,
            )
        elif gt_json is not None:
            gt_data = json.loads(Path(gt_json).read_text(encoding="utf-8"))
            gt_raw = {int(k) - 1: [int(x) - 1 for x in v] for k, v in gt_data.items()}
        else:
            raise ValueError("Provide ground_truth, config, gt_raw, or gt_json.")

    if explain_idxs is None:
        explain_idxs = sorted(int(k) for k in gt_raw.keys())

    if gt_raw_out is not None:
        gt_raw_out = Path(gt_raw_out)
        gt_raw_out.parent.mkdir(parents=True, exist_ok=True)
        gt_raw_serial = {str(int(k)): [int(x) for x in v] for k, v in gt_raw.items()}
        gt_raw_out.write_text(json.dumps(gt_raw_serial, indent=2), encoding="utf-8")

    if gt_out is not None:
        gt_out = Path(gt_out)
        gt_out.parent.mkdir(parents=True, exist_ok=True)
        gt_proc = {str(int(k) + 1): [int(x) + 1 for x in v] for k, v in gt_raw.items()}
        gt_out.write_text(json.dumps(gt_proc, indent=2), encoding="utf-8")

    df = pd.read_csv(ml_csv)
    idx_col = "idx" if "idx" in df.columns else ("e_idx" if "e_idx" in df.columns else None)
    if idx_col is None:
        raise ValueError("Processed CSV is missing 'idx'/'e_idx' column required for explain indices.")

    n = len(df)
    test_start = int(float(test_split) * n)
    test_idx_set = set(df.loc[test_start:, idx_col].astype(int).tolist())
    query_idxs = [int(k) + 1 for k in explain_idxs]
    query_idxs_test = [k for k in query_idxs if k in test_idx_set] or query_idxs

    cols = [idx_col, "u", "i", "ts"]
    if "label" in df.columns:
        cols.append("label")
    explain_df = df[df[idx_col].isin(query_idxs_test)][cols].sort_values(idx_col)
    explain_df = explain_df.rename(columns={idx_col: "idx"})
    explain_df.insert(0, "event_idx", explain_df["idx"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    explain_df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv, " (#items:", len(explain_df), ")")
    return out_csv




if __name__ == "__main__":
    main()

    DATASET = "stick_figure"

    ML_CSV = Path("data") / f"ml_{DATASET}.csv"
    GT_JSON = Path("data") / f"{DATASET}_gt.json"   # keys = query idxs (processed indexing)

    OUT = Path("resources/explainer/explain_index") / f"{DATASET}.csv"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    write_stick_figure_explain_index(
        ML_CSV,
        OUT,
        gt_json=GT_JSON,
        test_split=0.85,
        overwrite=True,
    )
