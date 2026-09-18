# -*- coding: utf-8 -*-
"""峰事件重组（推理结构优化）：谷底抬升合并 + 信号足点重划。

背景（2026-09-07 隔离分析）：开叉峰/宽峰的 GT 已改为"单包络框 + 信号足点边界"口径，
而模型按训练分布（谷底拆分标注）输出"子峰框集合"，且子峰 score 普遍低于主阈值；
推理管线在阈值过滤后没有任何事件级重组逻辑，导致该类峰系统性漏检（典型：test2_1
丙硫多菌灵-1，右子峰 0.94 保留、左子峰 0.56 被 0.9 阈值丢弃，GT 8.273-9.051 判 MISS）。

本模块在 predictor 的阈值过滤环节介入（run_single 调用）：
  1. 模型推理阶段以低分地板（event_merge_score_floor）保留全部候选框；
  2. 同一张 ROI 图内按"谷底抬升"合并相邻框：两框 apex 之间谷底最低值相对较矮
     峰高的抬升比例 > fork_frac（谷没有落下去，判为同一峰事件的叉）→ 合并为
     包络 [min_start, max_end]，score 取成员最大值；谷足够深（≤ fork_frac）
     则判为双峰/三峰，保持拆分；
  3. 对发生合并的事件按"信号足点"重划边界：从事件 apex 向两侧走到
     y < max(baseline+noise_k·noise, foot_frac·apex) 为止（可外扩也可内收，
     模型框伸进噪声平台的尾部会被收回），单侧不超过 max_extend，且不越过
     相邻事件与 ROI 窗口边界；未合并的正常峰保持模型原边界；
  4. 按 final_threshold 过滤事件 score，重建像素坐标框替换原 results。

所有 RT 计算基于 xic_matrix 的 RT 轴（分钟，可为全谱），像素↔RT 换算使用
roi_windows（xic_matrix 的 RT 轴是全谱，禁止用 t[0]/t[-1] 当窗口）。
"""
import os
import re

import numpy as np

from utils.roi_rt_mapping import ROI_IMAGE_WIDTH_PX, ROI_IMAGE_HEIGHT_PX

_N_PATTERN = re.compile(r"^(\d+)_mz", re.IGNORECASE)


def _xic_row_for_image(image_name, xic_count):
    """N_mz*.jpeg 命名 → XIC 行索引（N 为 1-based）；无法解析返回 -1。"""
    m = _N_PATTERN.match(os.path.basename(str(image_name)).strip())
    if not m:
        return -1
    idx = int(m.group(1)) - 1
    return idx if 0 <= idx < xic_count else -1


def _smooth(y, k=5):
    """轻量滑动平均，稳住 apex/谷底/足点判定；长度不足时原样返回。"""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < k or k < 2:
        return y
    kernel = np.ones(int(k), dtype=float) / float(k)
    return np.convolve(y, kernel, mode="same")


def _finite(t, y):
    m = np.isfinite(t) & np.isfinite(y)
    return t[m], y[m]


def _apex_index(t, y, lo, hi):
    """[lo, hi] RT 区间内 y 最大值索引；区间无点返回 -1。"""
    mask = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if not mask.any():
        return -1
    idx = np.where(mask)[0]
    sub = y[idx]
    sub = np.where(np.isnan(sub), -np.inf, sub)
    return int(idx[int(np.argmax(sub))])


def _baseline_and_noise(y):
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return 0.0, 0.0
    baseline = float(np.percentile(finite, 20))
    bottom = finite[finite <= np.percentile(finite, 30)]
    noise = float(np.std(bottom)) if bottom.size > 1 else 0.0
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(finite)) * 0.01
    return baseline, max(noise, 1e-12)


def _nms_dedup(events, contain_ratio=0.6):
    """同模态去重：按分数降序保留，若与已保留框的重叠 ≥ 较短框的 contain_ratio
    （即低分框基本是高分框的溢出/重检），丢弃低分框。返回新列表。"""
    kept = []
    for ev in sorted(events, key=lambda e: -e["score"]):
        dropped = False
        for k in kept:
            inter = min(ev["rt_max"], k["rt_max"]) - max(ev["rt_min"], k["rt_min"])
            width = max(min(ev["rt_max"] - ev["rt_min"], k["rt_max"] - k["rt_min"]), 1e-9)
            if inter / width >= contain_ratio:
                dropped = True
                break
        if not dropped:
            kept.append(ev)
    return sorted(kept, key=lambda e: e["rt_min"])


def _merge_events(events, t, y, fork_frac, diagnostics=None):
    """相邻事件按"谷底抬升"合并（排序后从左到右，合并结果继续向左尝试吞并）。

    判据：两框 apex 之间谷底最低值 / 较矮峰高 > fork_frac → 同一峰事件（叉）；
    否则谷足够深 → 双峰，保持拆分。
    events: [{rt_min, rt_max, score, merged}]；返回同结构列表。
    """
    merged = []
    for ev in sorted(events, key=lambda e: e["rt_min"]):
        cur = dict(ev)
        while merged:
            last = merged[-1]
            ia = _apex_index(t, y, last["rt_min"], last["rt_max"])
            ib = _apex_index(t, y, cur["rt_min"], cur["rt_max"])
            if ia < 0 or ib < 0:
                break
            lo, hi = (ia, ib) if ia <= ib else (ib, ia)
            seg = y[lo:hi + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                break
            valley = float(seg.min())
            h = float(min(y[ia], y[ib]))
            if not np.isfinite(h) or h <= 0:
                break
            ratio = valley / h
            if ratio <= fork_frac:
                break  # 谷足够深：判为分离的双峰，不合并
            if diagnostics is not None:
                diagnostics.append({
                    "rt_min": round(min(last["rt_min"], cur["rt_min"]), 3),
                    "rt_max": round(max(last["rt_max"], cur["rt_max"]), 3),
                    "valley_ratio": round(ratio, 3),
                    "scores": f"{last['score']:.2f}+{cur['score']:.2f}",
                })
            cur = {
                "rt_min": min(last["rt_min"], cur["rt_min"]),
                "rt_max": max(last["rt_max"], cur["rt_max"]),
                "score": max(last["score"], cur["score"]),
                "merged": True,
                "y1": min(last.get("y1", 0.0), cur.get("y1", 0.0)),
                "y2": max(last.get("y2", float("inf")), cur.get("y2", float("inf"))),
            }
            merged.pop()
        merged.append(cur)
    return merged


def _foot_rebound(events, t, y, floor_thresh, foot_frac, max_extend,
                  clamp_lo, clamp_hi):
    """对所有事件按信号足点重划边界（可外扩可内收）。

    足点判据：y < max(floor_thresh, foot_frac·事件apex) 即停
    （floor_thresh=baseline+noise_k·noise 兜底；foot_frac 相对事件峰高，
     与 GT"信号陡起/回落足点"口径对齐——实测足点约在峰高 0.5-1.5% 处）。
    从 apex 向两侧走，单侧总位移不超过 max_extend，且不越过相邻事件与窗口。
    """
    for i, ev in enumerate(events):
        lo_lim = max(events[i - 1]["rt_max"] if i > 0 else -np.inf, clamp_lo)
        hi_lim = min(events[i + 1]["rt_min"] if i + 1 < len(events) else np.inf, clamp_hi)
        orig_lo, orig_hi = ev["rt_min"], ev["rt_max"]
        ia = _apex_index(t, y, orig_lo, orig_hi)
        if ia < 0:
            continue
        thresh = max(floor_thresh, foot_frac * float(y[ia]))
        j = ia
        while (j - 1 >= 0 and t[j - 1] >= lo_lim
               and (orig_lo - t[j - 1]) <= max_extend + 1e-9
               and y[j - 1] >= thresh):
            j -= 1
        k = ia
        while (k + 1 < len(t) and t[k + 1] <= hi_lim
               and (t[k + 1] - orig_hi) <= max_extend + 1e-9
               and y[k + 1] >= thresh):
            k += 1
        ev["rt_min"] = float(t[j])
        ev["rt_max"] = float(t[k])


def _rt_to_px(rt, rt_lo, rt_hi, img_width):
    frac = (rt - rt_lo) / max(rt_hi - rt_lo, 1e-9)
    return float(np.clip(frac, 0.0, 1.0)) * img_width


def reorganize_events(results, xic_list, final_threshold, roi_windows=None,
                      fork_frac=0.35, max_extend=0.35, noise_k=2.0,
                      foot_frac=0.02, foot_extend=True,
                      img_width=ROI_IMAGE_WIDTH_PX, verbose=True,
                      diagnostics=None):
    """峰事件重组主入口（predictor.run_single 调用）。

    results: build_predictor 输出（像素框，已按 score_floor 保留候选）；
    xic_list: [ [rt_min 数组, 强度数组], ... ]（与 feature 行同序；RT 轴可为全谱）；
    roi_windows: {image_name: (rt_lo, rt_hi)} —— ROI 图像素↔RT 映射基准
    （xic_matrix 的 RT 轴是全谱，像素换算必须用本窗口，不能用 t[0]/t[-1]）；
    final_threshold: 重组后的事件分数过滤阈值（即用户 --threshold）。

    返回 (new_results, qc_rows)：qc_rows 与 qc3_threshold 表同 schema，
    口径为最终阈值（n_queries=候选框数）。
    """
    roi_windows = roi_windows or {}
    new_results = []
    qc_rows = []
    n_cand = n_before = n_after = n_merged_events = 0
    for res in results:
        img_path = res.get("image_path", "")
        image_name = os.path.basename(str(img_path)).strip()
        boxes = np.asarray(res.get("boxes", []), dtype=float).reshape(-1, 4)
        scores = np.asarray(res.get("scores", []), dtype=float).reshape(-1)
        n_cand += len(scores)

        idx = _xic_row_for_image(image_name, len(xic_list))
        win = roi_windows.get(image_name)
        if idx < 0 or len(scores) == 0 or win is None:
            # 无法对齐 XIC/窗口：退化为直接按最终阈值过滤，保持原行为
            keep = scores >= final_threshold
            if keep.any():
                new_results.append({
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "image_path": img_path,
                })
            qc_rows.append({
                "image": image_name,
                "n_queries": int(len(scores)),
                "n_kept": int(keep.sum()),
                "n_dropped": int(len(scores) - keep.sum()),
                "max_confidence": float(scores.max()) if len(scores) else None,
            })
            continue

        rt_lo, rt_hi = float(win[0]), float(win[1])
        xic = xic_list[idx]
        t = np.asarray(xic[0], dtype=float)
        y = _smooth(np.asarray(xic[1], dtype=float))
        t, y = _finite(t, y)
        n_before += len(scores)
        win_mask = (t >= rt_lo - 1e-9) & (t <= rt_hi + 1e-9)
        if t.size < 3 or not win_mask.any():
            keep = scores >= final_threshold
            if keep.any():
                new_results.append({"boxes": boxes[keep], "scores": scores[keep],
                                    "image_path": img_path})
            continue

        events = []
        for b, s in zip(boxes, scores):
            rt_a = rt_lo + (float(b[0]) / img_width) * (rt_hi - rt_lo)
            rt_b = rt_lo + (float(b[2]) / img_width) * (rt_hi - rt_lo)
            events.append({"rt_min": min(rt_a, rt_b), "rt_max": max(rt_a, rt_b),
                           "score": float(s), "merged": False,
                           "y1": float(b[1]), "y2": float(b[3])})

        # 同模态 NMS 去重 → 开叉合并 → 全事件足点重划 → 阈值过滤
        events = _nms_dedup(events)
        merged = _merge_events(events, t, y, fork_frac, diagnostics)
        if foot_extend:
            baseline, noise = _baseline_and_noise(y[win_mask])
            floor_thresh = baseline + noise_k * noise
            _foot_rebound(merged, t, y, floor_thresh, foot_frac,
                          max_extend, rt_lo, rt_hi)
        n_merged_events += sum(1 for e in merged if e["merged"])

        kept = [e for e in merged if e["score"] >= final_threshold]
        n_after += len(kept)
        if kept:
            new_boxes = np.empty((len(kept), 4), dtype=float)
            new_scores = np.empty((len(kept),), dtype=float)
            for i, e in enumerate(kept):
                x1 = _rt_to_px(e["rt_min"], rt_lo, rt_hi, img_width)
                x2 = _rt_to_px(e["rt_max"], rt_lo, rt_hi, img_width)
                new_boxes[i] = [min(x1, x2), e.get("y1", 0.0), max(x1, x2),
                                e.get("y2", ROI_IMAGE_HEIGHT_PX)]
                new_scores[i] = e["score"]
            new_results.append({"boxes": new_boxes, "scores": new_scores,
                                "image_path": img_path})
        qc_rows.append({
            "image": image_name,
            "n_queries": int(len(scores)),
            "n_kept": int(len(kept)),
            "n_dropped": int(len(scores) - len(kept)),
            "max_confidence": float(max((e["score"] for e in merged), default=0.0)),
        })

    if verbose:
        print(f"[INFO] 峰事件重组: 候选框 {n_cand} → 事件 {n_before} → 过阈值 {n_after} "
              f"(合并事件 {n_merged_events}，fork_frac={fork_frac}, "
              f"足点重划={'开' if foot_extend else '关'}, foot_frac={foot_frac}, "
              f"max_extend={max_extend}min, noise_k={noise_k})")
    return new_results, qc_rows
