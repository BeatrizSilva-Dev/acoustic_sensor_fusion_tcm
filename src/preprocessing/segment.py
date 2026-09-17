"""
Segments standardized audio into individual holes, with automatic detection of broken holes ("jam"),
synchronizing the holes detected in the reference microphone with the remaining microphones (including ultrasonic ones).
Updated: synchronization A (detect on the reference microphone and align the other microphones), 5 ms tolerance.
Modification: hole counter per microphone (file) resets for each drill.
"""

import os
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

# DIRECTORY CONFIGURATION
STANDARDIZED_DIR = "data/standardized"
SEGMENTED_DIR = "data/segmented"
DOCS_IMG_DIR = "docs/img/spectrograms"
METADATA_DIR = "data/metadata"
METADATA_CSV = os.path.join(METADATA_DIR, "segmented_metadata.csv")

# GLOBAL PARAMETERS
MIN_HOLE_DURATION = 2.0
SMOOTH_WINDOW = 11
VALLEY_WINDOW_SEC = 1.6
DEPTH_THRESH = 0.10
MIN_PROMINENCE_VALLEY = 0.005

HOP_LENGTH = 512
FRAME_LENGTH = 1024

GROUP_GAP_SEC = 2.5
MERGE_GAP_SEC = 2.0

DROP_PROMINENCE = 0.015
DROP_SEARCH_SEC = 1.5
START_FACTOR = 0.10 
START_ABS_THRESH = 0.02
START_SEARCH_SEC = 20.0

REFINE_DURATION_FACTOR = 1.3
REFINE_AGGRESSIVE_FACTOR = 2.5

MERGE_SHORT_ZTHRESH = 0.1

ALIGN_TOLERANCE_MS = 5.0
ALIGN_TOLERANCE_SEC = ALIGN_TOLERANCE_MS / 1000.0


# UTILITIES
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_jams_text(path):
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()

    text = text.replace(",", "\n")

    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r'(\d+)', line)
        if m:
            values.append(int(m.group(1)))

    jams = []
    current = None
    for v in values:
        if current is None:
            current = v
        else:
            current += v
        jams.append(current)

    return jams

def load_jams_for_drill(drill_folder_path):
    p = os.path.join(drill_folder_path, "jams.txt")
    if os.path.exists(p):
        jams = parse_jams_text(p)
        print(f" Jams file loaded: {p} -> {jams}")
        return jams
    return []

def find_datalogger(drill_path, drill_name):
    candidates = [
        os.path.join(drill_path, "datalogger.csv"),
        os.path.join(METADATA_DIR, "datalogger.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                time_col = next((c for c in df.columns if any(k in c.lower() for k in ("time","timestamp","ts","t","sec","s"))), None)
                volt_col = next((c for c in df.columns if "volt" in c.lower()), None)
                curr_col = next((c for c in df.columns if "curr" in c.lower() or "amp" in c.lower() or "i[" in c.lower()), None)
                return df, time_col, volt_col, curr_col
            except:
                return None, None, None, None
    return None, None, None, None

# HOLE DETECTION 
def detect_holes_by_deep_valleys(y, sr):
    hop_length = HOP_LENGTH
    frame_length = FRAME_LENGTH
    END_MARGIN_SEC = 0.0

    if np.max(np.abs(y)) == 0:
        return [], np.array([]), np.array([]), [], [], [], [], hop_length

    y = librosa.effects.preemphasis(y)
    y = y / max(1e-9, np.max(np.abs(y)))

    duration = len(y) / sr
    if duration > 40:
        fade_len = min(int(120 * sr), len(y))
        boost = np.linspace(3.0, 1.0, fade_len)
        y[:fade_len] *= boost

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = (
        np.convolve(rms, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode='same')
        if np.max(rms) > 0 else rms.copy()
    )
    rms_norm = rms_smooth / max(1e-9, np.max(rms_smooth))

    diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    if np.max(diff) > 0:
        diff /= np.max(diff)

    valleys_all, _ = signal.find_peaks(-rms_norm, prominence=MIN_PROMINENCE_VALLEY)

    window_frames = max(1, int((VALLEY_WINDOW_SEC * sr) / hop_length))
    valleys_kept = []
    depths = []
    local_peaks = {}

    for v in valleys_all:
        left_idx = max(0, v - window_frames)
        right_idx = min(len(rms_norm) - 1, v + window_frames)

        max_before = np.max(rms_norm[left_idx:v+1]) if v - left_idx >= 0 else rms_norm[v]
        max_after = np.max(rms_norm[v:right_idx+1]) if right_idx - v >= 0 else rms_norm[v]
        peak = max(max_before, max_after)

        depth = peak - rms_norm[v]
        depths.append(depth)
        local_peaks[v] = peak

        if depth >= DEPTH_THRESH:
            valleys_kept.append(v)

    if len(valleys_kept) < 2 and len(valleys_all) >= 2:
        N = min(10, len(valleys_all))
        sorted_idx = np.argsort(depths)[-N:]
        valleys_kept = list(np.array(valleys_all)[sorted_idx])
        valleys_kept.sort()

    drop_peaks, _ = signal.find_peaks(-diff, prominence=DROP_PROMINENCE)

    holes = []
    if valleys_kept:
        group_start = valleys_kept[0]
        group_end = valleys_kept[0]
        max_gap = int((GROUP_GAP_SEC * sr) / hop_length)

        for i in range(1, len(valleys_kept)):
            if valleys_kept[i] - valleys_kept[i - 1] > max_gap:
                dur = (group_end - group_start) * hop_length / sr
                if dur >= MIN_HOLE_DURATION:
                    holes.append((group_start * hop_length, group_end * hop_length))
                group_start = valleys_kept[i]
            group_end = valleys_kept[i]

        dur = (group_end - group_start) * hop_length / sr
        if dur >= MIN_HOLE_DURATION:
            holes.append((group_start * hop_length, group_end * hop_length))

    merged_holes = []
    for s, e in holes:
        if not merged_holes:
            merged_holes.append((s, e))
            continue
        last_s, last_e = merged_holes[-1]
        if (s/sr) - (last_e/sr) < MERGE_GAP_SEC:
            merged_holes[-1] = (last_s, e)
        else:
            merged_holes.append((s, e))
    holes = merged_holes

    refined_holes = []
    search_window_drop = int(DROP_SEARCH_SEC * sr / hop_length)
    search_window_start = int(START_SEARCH_SEC * sr / hop_length)
    end_margin_frames = int(END_MARGIN_SEC * sr / hop_length)

    for idx, (s, e) in enumerate(holes):
        s_frame = int(s / hop_length)
        e_frame = int(e / hop_length)
        s_new_frame = None

        if valleys_kept:
            candidates = [v for v in valleys_kept if v <= s_frame + 2]
            if candidates:
                ref_valley = candidates[-1]
            else:
                ref_valley = valleys_kept[np.argmin(np.abs(np.array(valleys_kept) - s_frame))]
            local_peak = local_peaks.get(ref_valley, None)
        else:
            local_peak = None

        if local_peak is not None:
            start_threshold = max(START_ABS_THRESH, local_peak * START_FACTOR)
            start_left = max(0, s_frame - search_window_start)
            segment = rms_norm[start_left:s_frame]
            if segment.size > 0:
                hits = np.where(segment >= start_threshold)[0]
                if hits.size > 0:
                    s_new_frame = start_left + hits[0]

        if s_new_frame is None:
            candidates_drop = [p for p in drop_peaks if (s_frame - search_window_drop) <= p < s_frame]
            if candidates_drop:
                s_new_frame = candidates_drop[-1]

        if idx == 0:
            s_new_frame = 0

        if s_new_frame is not None:
            s = int(s_new_frame * hop_length)

        if refined_holes:
            s = max(s, refined_holes[-1][1])

        e_ext = min(len(y), int((e_frame + end_margin_frames) * hop_length))
        if idx == len(holes) - 1:
            e_ext = len(y)

        refined_holes.append((s, e_ext))

    return refined_holes, rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length




def merge_small_outlier_holes(holes, sr, k):
    """
    Merges only holes that are significantly smaller than the average (outliers),
    based on mean - k * standard deviation.

    holes: list of (start, end) in samples
    sr: sampling rate
    k: filter intensity (1.0 = moderate, 1.5 = more restrictive)
    """
    if not holes or len(holes) < 2:
        return holes

    # computes durations in seconds
    durations = np.array([(e - s) / sr for s, e in holes])
    mean_dur = durations.mean()
    std_dur = durations.std()

    threshold = mean_dur - k * std_dur

    merged = []
    for i, (s, e) in enumerate(holes):
        dur = durations[i]

        if dur < threshold and len(merged) > 0:
            prev_s, prev_e = merged[-1]
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    return merged


def detect_holes_by_deep_valleys_custom(y, sr, DEPTH_THRESH_param=0.15, GROUP_GAP_SEC_param=3.0, MERGE_GAP_SEC_param=2.0):
    """
    Calls detect_holes_by_deep_valleys while temporarily overriding some global parameters.
    Returns the same output as detect_holes_by_deep_valleys.
    """
    orig_DEPTH = globals().get('DEPTH_THRESH', DEPTH_THRESH)
    orig_GROUP = globals().get('GROUP_GAP_SEC', GROUP_GAP_SEC)
    orig_MERGE = globals().get('MERGE_GAP_SEC', MERGE_GAP_SEC)

    globals()['DEPTH_THRESH'] = DEPTH_THRESH_param
    globals()['GROUP_GAP_SEC'] = GROUP_GAP_SEC_param
    globals()['MERGE_GAP_SEC'] = MERGE_GAP_SEC_param

    results = detect_holes_by_deep_valleys(y, sr)

    globals()['DEPTH_THRESH'] = orig_DEPTH
    globals()['GROUP_GAP_SEC'] = orig_GROUP
    globals()['MERGE_GAP_SEC'] = orig_MERGE

    return results

def refine_long_holes_sensive(
        y,
        sr,
        holes,
        duration_factor,
        aggressive_factor
):
    """
    Refines abnormally long holes using more sensitive recursive detection.

    duration_factor:
        A hole longer than (duration_factor × median) is selected for refinement.

    aggressive_factor:
        A hole longer than (aggressive_factor × median) enters aggressive refinement mode.
    """

    if not holes:
        return holes

    durations = np.array([(e - s) / sr for s, e in holes])
    median_dur = np.median(durations) if durations.size > 0 else 0.0

    refined_holes = []

    MIN_SUB_HOLE_SEC = 0.05
    RMS_OSC_RATIO = 0.15    

    for (s, e), dur in zip(holes, durations):

        y_seg = y[s:e]

        rms = librosa.feature.rms(
            y=y_seg,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )[0]

        rms_mean = np.mean(rms) if rms.size > 0 else 0
        rms_std = np.std(rms) if rms.size > 0 else 0
        rms_oscillating = (
                rms_mean > 0 and (rms_std / rms_mean) > RMS_OSC_RATIO
        )

        must_refine = (
                median_dur > 0 and (
                dur > duration_factor * median_dur or
                rms_oscillating
        )
        )

        if not must_refine:
            refined_holes.append((s, e))
            continue

        print(
            f"→ Refining hole ({dur:.2f}s | "
            f"{'Oscillating RMS' if rms_oscillating else 'long'})"
        )

        if median_dur > 0 and dur > aggressive_factor * median_dur:
            params = dict(
                DEPTH_THRESH_param=0.035,
                GROUP_GAP_SEC_param=0.35,
                MERGE_GAP_SEC_param=0.15
            )
        else:
            params = dict(
                DEPTH_THRESH_param=0.06,
                GROUP_GAP_SEC_param=0.7,
                MERGE_GAP_SEC_param=0.35
            )

        sub_holes, *_ = detect_holes_by_deep_valleys_custom(
            y_seg,
            sr,
            **params
        )

        sub_holes_global = [
            (s + s2, s + e2)
            for (s2, e2) in sub_holes
            if (e2 - s2) / sr >= MIN_SUB_HOLE_SEC
        ]

        if len(sub_holes_global) == 0:
            refined_holes.append((s, e))
        else:
            refined_holes.extend(sub_holes_global)

    return refined_holes



def refine_long_holes(y, sr, holes, duration_factor=REFINE_DURATION_FACTOR, aggressive_factor=REFINE_AGGRESSIVE_FACTOR):
    """
    Refines abnormally long holes using more sensitive recursive detection.

    duration_factor: threshold (× median) above which a hole is refined.
    aggressive_factor: if the hole is much longer than the threshold, even more sensitive parameters are applied.
    """
    if not holes:
        return holes

    durations = np.array([(e - s) / sr for s, e in holes])
    median_dur = np.median(durations) if durations.size > 0 else 0.0
    refined_holes = []

    for (s, e), dur in zip(holes, durations):
        if median_dur == 0 or dur <= duration_factor * median_dur:
            refined_holes.append((s, e))
            continue

        print(f"Refining long hole ({dur:.2f}s)..")

        y_seg = y[s:e]

        if dur > aggressive_factor * median_dur:
            params = dict(DEPTH_THRESH_param=0.05, GROUP_GAP_SEC_param=0.5, MERGE_GAP_SEC_param=0.25)
        else:
            params = dict(DEPTH_THRESH_param=0.08, GROUP_GAP_SEC_param=1.0, MERGE_GAP_SEC_param=0.5)

        sub_holes, *_ = detect_holes_by_deep_valleys_custom(y_seg, sr, **params)
        sub_holes_global = [(s + s2, s + e2) for (s2, e2) in sub_holes]

        if len(sub_holes_global) == 0:
            refined_holes.append((s, e))
        else:
            refined_holes.extend(sub_holes_global)

    return refined_holes

def expand_holes_backward(holes):
    """
    Expands the start of each hole (starting from the second one)
    up to the end of the previous hole, ensuring that no
    gap remains between them.

    holes: list of (start_sample, end_sample)
    """
    if not holes or len(holes) < 2:
        return holes

    holes = sorted(holes, key=lambda x: x[0])

    expanded = [holes[0]]  

    for i in range(1, len(holes)):
        prev_s, prev_e = expanded[-1]
        s, e = holes[i]

        expanded.append((prev_e, e))

    return expanded


def expand_holes_to_next(holes):
    """
    Expands the end of each hole up to the start of the next one,
    ensuring that no gap remains between them.

    holes: list of (start_sample, end_sample)
    """
    if not holes or len(holes) < 2:
        return holes

    holes = sorted(holes, key=lambda x: x[0])

    expanded = []

    for i in range(len(holes) - 1):
        s, e = holes[i]
        next_s, _ = holes[i + 1]

        expanded.append((s, next_s))

    expanded.append(holes[-1])

    return expanded

def align_holes_to_reference(holes_ref, y_other, sr, tolerance_sec=ALIGN_TOLERANCE_SEC, hop_length=HOP_LENGTH, frame_length=FRAME_LENGTH):
    """
    Aligns the holes detected in the reference microphone with the "other" microphone.

    Strategy:
    - Computes the RMS of the "other" microphone using the same frame and hop lengths.
    - For each hole detected in the reference microphone, takes its center (in samples)
    and searches for the highest-energy RMS frame within a tolerance window (in frames).
    - Adjusts the start and end positions using half the duration of the reference hole
    (preserving proportionality), expanding them by the tolerance in samples.

    Returns:
        List of (start_sample, end_sample) in the "other" microphone sample domain.
    """
    if len(holes_ref) == 0:
        return []

    if isinstance(y_other, np.ndarray) and y_other.ndim == 2:
        mono = np.mean(y_other, axis=1)
    else:
        mono = np.array(y_other, dtype=float)

    rms_other = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=hop_length)[0]
    n_frames = len(rms_other)
    frames_time_samples = np.arange(n_frames) * hop_length  # position in samples for each frame

    tol_frames = max(1, int((tolerance_sec * sr) / hop_length))
    tol_samples = max(1, int(tolerance_sec * sr))

    holes_aligned = []
    prev_end = 0
    for (s_ref, e_ref) in holes_ref:
        center_ref = int((s_ref + e_ref) // 2)
        frame_idx = int(center_ref // hop_length)

        lo = max(0, frame_idx - tol_frames)
        hi = min(n_frames - 1, frame_idx + tol_frames)

        # if RMS is all zeros or very small, fallback to mapping by sample index directly
        if n_frames == 0 or np.all(rms_other <= 0):
            mapped_sample = center_ref
        else:
            search = rms_other[lo:hi+1]
            if search.size == 0:
                mapped_sample = center_ref
            else:
                best_rel = np.argmax(search)
                best_idx = lo + best_rel
                mapped_sample = int(frames_time_samples[best_idx])

        half_width = int((e_ref - s_ref) // 2)
        s_adj = max(0, mapped_sample - half_width - tol_samples)
        e_adj = min(len(mono), mapped_sample + half_width + tol_samples)

        if s_adj < prev_end:
            s_adj = prev_end
            if s_adj >= e_adj:
                e_adj = min(len(mono), s_adj + 1)

        prev_end = e_adj
        holes_aligned.append((int(s_adj), int(e_adj)))

    return holes_aligned


def merge_short_holes_by_recursion(holes, sr, factor):
    """
    Merges holes whose duration is significantly shorter than the median.

    Strategy:
    - Computes the median hole duration.
    - Finds the FIRST hole below the threshold.
    - Merges that hole with the NEXT one.
    - Re-evaluates the entire list.
    - Repeats until no spurious holes remain.
    """
    if not holes or len(holes) < 2:
        return holes

    holes = sorted(holes, key=lambda x: x[0])

    while True:
        durations = np.array([(e - s) / sr for s, e in holes])
        median_dur = np.median(durations)
        threshold = factor * median_dur

        merged_any = False
        new_holes = []
        i = 0

        while i < len(holes):
            s, e = holes[i]
            dur = (e - s) / sr

            if dur < threshold and i + 1 < len(holes):
                ns, ne = holes[i + 1]

                new_holes.append((s, ne))
                i += 2
                merged_any = True
                break
            else:
                new_holes.append((s, e))
                i += 1

        if i < len(holes):
            new_holes.extend(holes[i:])

        holes = new_holes

        if not merged_any:
            break

    return holes

def merge_holes_by_index(holes, idx1, idx2):
    """
    Merges hole idx1 with hole idx2 into a single hole.

    holes: list of (start_sample, end_sample)
    idx1, idx2: indices of the holes to be merged
    """
    if idx1 == idx2:
        return holes

    n = len(holes)
    if not (0 <= idx1 < n and 0 <= idx2 < n):
        raise IndexError("Hole index out of range")

    i, j = sorted([idx1, idx2])

    s1, e1 = holes[i]
    s2, e2 = holes[j]

    new_hole = (min(s1, s2), max(e1, e2))

    new_holes = [
        h for k, h in enumerate(holes)
        if k not in (i, j)
    ]

    new_holes.append(new_hole)
    new_holes.sort(key=lambda x: x[0])

    return new_holes

def add_initial_hole_and_reduce(holes, sr, duration_sec):
    """
    Adds a hole at the beginning by consuming part of the duration
    of the only existing hole.

    Expects exactly one hole as input.

    holes: [(start_sample, end_sample)]
    sr: sample rate (Hz)
    duration_sec: duration of the new initial hole (seconds)
    """
    if len(holes) != 1:
        raise ValueError("Function expects exactly one hole")

    shift = int(duration_sec * sr)

    s, e = holes[0]
    original_duration = e - s

    if shift >= original_duration:
        return [(0, original_duration)]

    new_holes = []

    new_holes.append((0, shift))

    new_start = s + shift
    new_holes.append((new_start, e))

    return new_holes

def adjust_hole_boundary(holes, sr, hole_idx, side, delta_sec):
    """
    Manually adjusts the start or end of a hole,
    automatically compensating the neighboring hole.

    holes: list of (start_sample, end_sample)
    sr: sample rate
    hole_idx: index of the hole to adjust
    side: "start" or "end"
    delta_sec: value in seconds
        > 0 → increases
        < 0 → decreases
    """
    if hole_idx < 0 or hole_idx >= len(holes):
        raise IndexError("hole_idx out of range")

    delta_samples = int(delta_sec * sr)
    holes = list(holes)  

    s, e = holes[hole_idx]

    if side == "end":
        if hole_idx + 1 >= len(holes):
            raise ValueError("There is no next hole to compensate")

        ns, ne = holes[hole_idx + 1]

        new_e = e + delta_samples
        new_ns = ns + delta_samples

        if new_e <= s:
            raise ValueError("Hole would have an invalid duration")
        if new_ns >= ne:
            raise ValueError("Next hole would become invalid")

        holes[hole_idx] = (s, new_e)
        holes[hole_idx + 1] = (new_ns, ne)

    elif side == "start":
        if hole_idx - 1 < 0:
            raise ValueError("There is no previous hole to compensate")

        ps, pe = holes[hole_idx - 1]

        new_s = s + delta_samples
        new_pe = pe + delta_samples

        if new_s >= e:
            raise ValueError("Hole would have an invalid duration")
        if new_pe <= ps:
            raise ValueError("Previous hole would become invalid")

        holes[hole_idx] = (new_s, e)
        holes[hole_idx - 1] = (ps, new_pe)

    else:
        raise ValueError("side must be 'start' or 'end'")

    return holes

def adjust_hole_boundaries_by_ids(holes, sr, hole_ids, side, delta_sec):
    """
    Manually adjusts the start or end of multiple holes,
    automatically compensating the corresponding neighboring hole.

    holes: list of (start_sample, end_sample)
    sr: sample rate
    hole_ids: list of indices of the holes to adjust
    side: "start" or "end"
    delta_sec: value in seconds
        > 0 → increases
        < 0 → decreases

    Example:
        adjust_hole_boundaries_by_ids(
            holes,
            sr=44100,
            hole_ids=[1, 3, 5],
            side="end",
            delta_sec=0.2
        )
    """
    holes = list(holes) 
    delta_samples = int(delta_sec * sr)

    if side == "end":
        ordered_ids = sorted(hole_ids, reverse=True)

        for hole_idx in ordered_ids:
            if hole_idx < 0 or hole_idx >= len(holes):
                raise IndexError(f"hole_idx {hole_idx} out of range")

            if hole_idx + 1 >= len(holes):
                raise ValueError(
                    f"Hole {hole_idx} has no subsequent hole for compensation"
                )

            s, e = holes[hole_idx]
            ns, ne = holes[hole_idx + 1]

            new_e = e + delta_samples
            new_ns = ns + delta_samples

            if new_e <= s:
                raise ValueError(
                    f"Hole {hole_idx} would have an invalid duration"
                )

            if new_ns >= ne:
                raise ValueError(
                    f"The next hole after index {hole_idx} would become invalid"
                )

            holes[hole_idx] = (s, new_e)
            holes[hole_idx + 1] = (new_ns, ne)

    elif side == "start":
        ordered_ids = sorted(hole_ids)

        for hole_idx in ordered_ids:
            if hole_idx < 0 or hole_idx >= len(holes):
                raise IndexError(f"hole_idx {hole_idx} out of range")

            if hole_idx - 1 < 0:
                raise ValueError(
                    f"Hole {hole_idx} does not have a previous hole for compensation"
                )

            s, e = holes[hole_idx]
            ps, pe = holes[hole_idx - 1]

            new_s = s + delta_samples
            new_pe = pe + delta_samples

            if new_s >= e:
                raise ValueError(
                    f"Hole {hole_idx} would have an invalid duration"
                )

            if new_pe <= ps:
                raise ValueError(
                    f"Previous hole at index {hole_idx} would become invalid"
                )

            holes[hole_idx] = (new_s, e)
            holes[hole_idx - 1] = (ps, new_pe)

    else:
        raise ValueError("side must be 'start' or 'end'")

    return holes



def merge_short_holes_by_median(holes, sr, factor):
    """
    Merges holes with duration far below the median.

    factor = 0.4 → hole < 40% of median duration is considered spurious
    """
    if not holes or len(holes) < 2:
        return holes

    durations = np.array([(e - s) / sr for s, e in holes])
    median_dur = np.median(durations)

    threshold = factor * median_dur

    merged = []
    i = 0
    while i < len(holes):
        s, e = holes[i]
        dur = (e - s) / sr

        if dur < threshold:
            if merged:
                ps, pe = merged[-1]
                merged[-1] = (ps, e)
            elif i + 1 < len(holes):
                ns, ne = holes[i + 1]
                merged.append((s, ne))
                i += 1
            else:
                merged.append((s, e))
        else:
            merged.append((s, e))

        i += 1

    return merged


def fix_internal_energy_split(
        holes,
        rms_norm,
        diff,
        sr,
        hop_length,
        min_rel_drop,
        search_ratio
):
    """
    Fixes cases where the start of hole N contains energy from hole N-1,
    by identifying a real energy drop WITHIN the hole.

    Strategy:
    - Analyzes the start of each hole (except the first)
    - Looks for a consistent energy valley
    - Uses RMS + diff (same basis as the main detector)
    """

    if not holes or len(holes) < 2:
        return holes

    fixed = [holes[0]]

    for i in range(1, len(holes)):
        prev_s, prev_e = fixed[-1]
        cur_s, cur_e = holes[i]

        s_frame = int(cur_s / hop_length)
        e_frame = int(cur_e / hop_length)
        search_end = s_frame + int((e_frame - s_frame) * search_ratio)
        search_end = min(search_end, e_frame)

        if search_end <= s_frame + 2:
            fixed.append((cur_s, cur_e))
            continue

        rms_seg = rms_norm[s_frame:search_end]
        diff_seg = diff[s_frame:search_end]

        if rms_seg.size < 3:
            fixed.append((cur_s, cur_e))
            continue

        peak_energy = np.max(rms_seg)

        valley_idxs = np.where(rms_seg < peak_energy * (1 - min_rel_drop))[0]

        if valley_idxs.size == 0:
            fixed.append((cur_s, cur_e))
            continue

        valley_idx = valley_idxs[0]

        local_diff = diff_seg[max(0, valley_idx - 2): valley_idx + 2]
        if np.max(local_diff) < DROP_PROMINENCE:
            fixed.append((cur_s, cur_e))
            continue

        split_frame = s_frame + valley_idx
        split_sample = int(split_frame * hop_length)

        split_sample = max(prev_s + 1, split_sample)
        split_sample = min(cur_e - 1, split_sample)

        fixed[-1] = (prev_s, split_sample)
        fixed.append((split_sample, cur_e))

    return fixed

# DATALOGGER
def find_column_root(take_path):
    """
    Walks up the directory tree until it finds the column_X folder.
    Returns the absolute path of the column folder.
    """
    cur = os.path.abspath(take_path)

    while True:
        base = os.path.basename(cur).lower()
        if base.startswith("column_"):
            return cur

        parent = os.path.dirname(cur)
        if parent == cur:  
            return None

        cur = parent

import re

def load_datalogger_xls(datalogger_dir):
    if not datalogger_dir or not os.path.isdir(datalogger_dir):
        return None

    files = [f for f in os.listdir(datalogger_dir) if f.lower().endswith(".xls")]
    if not files:
        return None

    path = os.path.join(datalogger_dir, files[0])
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
    except Exception:
        return None

    text = None
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            text = raw_bytes.decode(enc)
            break
        except Exception:
            pass

    if text is None:
        return None

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    voltage_re = re.compile(r"([\d.,]+)\s*V", re.IGNORECASE)
    current_re = re.compile(r"([\d.,]+)\s*A", re.IGNORECASE)
    time_re = re.compile(
        r"(\d{2}/\d{2}/\d{2,4}\s+\d{2}:\d{2}:\d{2})"
    )

    records = []

    for line in lines:
        v = voltage_re.search(line)
        c = current_re.search(line)
        t = time_re.search(line)

        if not (v and c and t):
            continue

        try:
            voltage = float(v.group(1).replace(",", "."))
            current = float(c.group(1).replace(",", "."))
            time = pd.to_datetime(
                t.group(1),
                dayfirst=True,
                errors="coerce"
            )
        except Exception:
            continue

        if pd.isna(time):
            continue

        records.append({
            "voltage": voltage,
            "current": current,
            "time": time
        })

    if not records:
        return None

    return pd.DataFrame(records).reset_index(drop=True)

def extract_datalogger_window(
        datalog_df,
        hole_start_sec,
        hole_end_sec,
        audio_start_time
):
    """
    Returns lists of Voltage and Current corresponding to the hole's interval.

    audio_start_time: absolute datetime corresponding to t=0 of the audio
    """
    if datalog_df is None or datalog_df.empty:
        return [], []

    start_time = audio_start_time + pd.to_timedelta(hole_start_sec, unit="s")
    end_time   = audio_start_time + pd.to_timedelta(hole_end_sec, unit="s")

    mask = (datalog_df["time"] >= start_time) & (datalog_df["time"] <= end_time)
    window = datalog_df.loc[mask]

    if window.empty:
        return [], []

    return (
        window["voltage"].round(2).tolist(),
        window["current"].round(2).tolist()
    )


def plot_spectrogram_with_holes(y, sr, holes, output_path):
    plt.figure(figsize=(14, 6))
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram with detected holes')
    for i, (s, e) in enumerate(holes, 1):
        plt.axvspan(s / sr, e / sr, color='lime', alpha=0.3)
        plt.axvline(s / sr, color='white', linestyle='--', linewidth=0.6)
        plt.axvline(e / sr, color='white', linestyle='--', linewidth=0.6)
        mid = (s + e) / 2 / sr
        plt.text(mid, sr / 4, f'{i}', color='white', ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def force_temporal_split_holes(
        holes,
        sr,
        duration_factor,
        min_hole_sec=0.08,
        max_parts=8
):
    """
    Forces the temporal cut of abnormally long holes.

    holes: list of (start_sample, end_sample)
    sr: sample rate

    duration_factor:
        hole > duration_factor × median → will be split

    min_hole_sec:
        minimum acceptable duration of each sub-hole

    max_parts:
        upper limit of splits to avoid explosion
    """

    if not holes or len(holes) < 2:
        return holes
    
    durations = np.array([(e - s) / sr for s, e in holes])
    median_dur = np.median(durations)

    if median_dur <= 0:
        return holes

    refined = []

    for s, e in holes:
        dur = (e - s) / sr
        if dur <= duration_factor * median_dur:
            refined.append((s, e))
            continue

        n_parts = int(np.round(dur / median_dur))
        n_parts = max(2, min(n_parts, max_parts))

        print(
            f"Forced temporal cut: "
            f"{dur:.2f}s → {n_parts} parts"
        )

        step = (e - s) // n_parts

        for i in range(n_parts):
            ss = s + i * step
            ee = s + (i + 1) * step if i < n_parts - 1 else e

            if (ee - ss) / sr >= min_hole_sec:
                refined.append((ss, ee))

    return refined



def compute_holes_before_fail(hole_num, jams_list):
    """
    Returns how many holes remain until the next failure (jam).
    - 0  → the hole itself is a jam
    - n>0 → n holes remain until the jam
    - None → there is no future jam
    """
    future_jams = [j for j in jams_list if j >= hole_num]
    if not future_jams:
        return None
    return min(future_jams) - hole_num


def column_folder_key(name: str):
    col_match = re.search(r'column_(\d+)', name)
    col_num = int(col_match.group(1)) if col_match else 999

    if re.search(r'(^|_)no_jam(_|$)', name):
        ord_num = 999
    else:
        ord_match = re.search(r'_(\d+)(st|nd|rd|th)(_|$)', name)
        ord_num = int(ord_match.group(1)) if ord_match else 0

    return (col_num, ord_num)

def get_experiment_root(path):
    parts = os.path.normpath(path).split(os.sep)

    try:
        idx = parts.index("standardized")
        return os.sep.join(parts[:idx + 2])
    except ValueError:
        raise RuntimeError(f"Invalid path (standardized not found): {path}")

def process_take_folder(take_path, drill, jams_list, metadata, hole_counter,
                        standardized_root=STANDARDIZED_DIR,
                        segmented_root=SEGMENTED_DIR):

    wavs = [f for f in os.listdir(take_path) if f.lower().endswith(".wav")]
    if not wavs:
        return

    ref_path = os.path.join(take_path, wavs[0])
    try:
        y_ref, sr_ref = librosa.load(ref_path, sr=None, mono=True)
    except Exception as ex:
        print(f"Failed to load {ref_path}: {ex}")
        return

    column_root = find_column_root(take_path)

    datalogger_dir = None
    if column_root is not None:
        datalogger_dir = os.path.join(column_root, "datalogger")

    datalog_df = load_datalogger_xls(datalogger_dir)

    audio_start_time = None
    if datalog_df is not None and not datalog_df.empty:
        audio_start_time = datalog_df["time"].iloc[0]

    holes, rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length = \
        detect_holes_by_deep_valleys(y_ref, sr_ref)

    holes = merge_small_outlier_holes(holes, sr_ref, k=1.5)

    holes = refine_long_holes(
        y_ref, sr_ref, holes,
        duration_factor=REFINE_DURATION_FACTOR,
        aggressive_factor=REFINE_AGGRESSIVE_FACTOR
    )

    holes = merge_small_outlier_holes(holes, sr_ref, k=2.0)

    holes = expand_holes_to_next(holes)

    holes = merge_short_holes_by_median(holes, sr_ref, factor=0.4)

    if drill.lower() == 'drill_4mm_22_batch_03_collet_2_14-02-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")
        if holes == [(0, 5675520), (5675520, 10036224), (10036224, 14719488), (14719488, 19375616), (19375616, 24000512), (24000512, 28656640), (28656640, 33412608), (33412608, 37973504), (37973504, 42686976), (42686976, 47345664), (47345664, 52009984), (52009984, 56574976), (56574976, 61234688), (61234688, 65884160), (65884160, 70552064), (70552064, 75211264), (75211264, 79925760), (79925760, 84509184), (84509184, 89412096), (89412096, 94041600), (94041600, 98551808), (98551808, 103488986)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 2892800), (2892800, 5319680), (5319680, 7597056), (7597056, 9845248), (9845248, 12133888), (12133888, 14638592), (14638592, 16942080), (16942080, 19277824), (19277824, 21615104), (21615104, 23783424), (23783424, 26250752), (26250752, 28550656), (28550656, 30771200), (30771200, 33096704), (33096704, 35426304), (35426304, 37734912), (37734912, 40087552), (40087552, 42413568), (42413568, 44729856), (44729856, 47067648), (47067648, 49396736), (49396736, 51744000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                side="end",
                delta_sec=1.00
            )
        else:
            pass

    if drill.lower() == 'drill_4mm_03_batch_00_collet_1_30-01-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")

        if holes == [(0, 1300480), (1300480, 2429440), (2429440, 3721216), (3721216, 4999680), (4999680, 6056960), (6056960, 7223808), (7223808, 8414208), (8414208, 9551360), (9551360, 10711040), (10711040, 11899392), (11899392, 13021184), (13021184, 14206464), (14206464, 15417344), (15417344, 16581120), (16581120, 17649152), (17649152, 18910208), (18910208, 20072448), (20072448, 21234688), (21234688, 22397440), (22397440, 23543808), (23543808, 24727552), (24727552, 25888256), (25888256, 27033088), (27033088, 28201472), (28201472, 29335040), (29335040, 30562816), (30562816, 31713280), (31713280, 32828928), (32828928, 34017792), (34017792, 35424188)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 5844480), (5844480, 10557952), (10557952, 15190528), (15190528, 19936256), (19936256, 24529920), (24529920, 29186048), (29186048, 33772544), (33772544, 38104576), (38104576, 45703258)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=7,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 5590016), (5590016, 10364928), (10364928, 14921216), (14921216, 19562496), (19562496, 23576064), (23576064, 28803176)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 2322944), (2322944, 5259776), (5259776, 7585280), (7585280, 9882624), (9882624, 12215296), (12215296, 14543360), (14543360, 16871424), (16871424, 19205632), (19205632, 21524480), (21524480, 23860736), (23860736, 26288640), (26288640, 28518400), (28518400, 30828032), (30828032, 33290752), (33290752, 35493376), (35493376, 37826560), (37826560, 40151552), (40151552, 42463744), (42463744, 44808704), (44808704, 47130112), (47130112, 49465344), (49465344, 51896320), (51896320, 54119936), (54119936, 56442368), (56442368, 58772480), (58772480, 61096448), (61096448, 63420416), (63420416, 65759232), (65759232, 68086784), (68086784, 70848000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=6.00
            )
        elif holes == [(0, 2994688), (2994688, 5324800), (5324800, 7651328), (7651328, 9981440), (9981440, 12300800), (12300800, 14603776), (14603776, 16939008), (16939008, 19313152), (19313152, 22848000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=7,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 3037184), (3037184, 5291520), (5291520, 6287360), (6287360, 7664128), (7664128, 10036736), (10036736, 11920384), (11920384, 14400000)]:
            holes = merge_holes_by_index(holes, idx1=2, idx2=3)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=4.00
            )
        elif holes == []:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=30.0)
        else:
            pass
    if drill.lower() == 'drill_4mm_01_batch_00_collet_1_29-01-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")

        if holes == [(0, 1488896), (1488896, 2659328), (2659328, 3755520), (3755520, 4911104), (4911104, 6191786)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 3092992), (3092992, 5430784), (5430784, 7769600), (7769600, 10089984), (10089984, 12425216), (12425216, 14749696), (14749696, 17056768), (17056768, 19131392), (19131392, 23136000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=7,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 3577856), (3577856, 6192128), (6192128, 8905216), (8905216, 11277312), (11277312, 13593088), (13593088, 15930368), (15930368, 18203648), (18203648, 20554240), (20554240, 22896128), (22896128, 25234432), (25234432, 28217344), (28217344, 29821440), (29821440, 32207872), (32207872, 34520576), (34520576, 36852736), (36852736, 39195136), (39195136, 41495040), (41495040, 43844608), (43844608, 46142976), (46142976, 48484352), (48484352, 50778624), (50778624, 53163520), (53163520, 55478272), (55478272, 57831936), (57831936, 60113408), (60113408, 62447104), (62447104, 64813056), (64813056, 67090432), (67090432, 69432320), (69432320, 72384000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=10,
                side="end",
                delta_sec=-3.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_17_batch_01_collet_1_06-02-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")
        if holes == [(0, 4516864), (4516864, 10003968), (10003968, 14661632), (14661632, 19316736), (19316736, 24100352), (24100352, 28738560), (28738560, 37439002)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 5480960), (5480960, 10145792), (10145792, 18809192)] or holes == [(0, 5257216), (5257216, 9914880), (9914880, 17848230)] or holes == [(0, 2803712), (2803712, 5118976), (5118976, 9216000)] or holes == [(0, 2888192), (2888192, 5227008), (5227008, 8928000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 9020212)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=30.0)
        elif holes == [(0, 3402240), (3402240, 5286912), (5286912, 7596032), (7596032, 9927168), (9927168, 12265984), (12265984, 14590464), (14590464, 16923136), (16923136, 18720000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-4.00
            )
        else:
            pass

    if drill.lower() == 'drill_4mm_23_batch_01_collet_2_14-02-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")
        if holes == [(0, 4224000)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=31.0)
        elif holes == [(0, 8443636)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=30.0)
        elif holes == [(0, 2748928), (2748928, 5082624), (5082624, 7409152), (7409152, 9731072), (9731072, 11232000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5445632), (5445632, 9952256), (9952256, 14604288), (14604288, 19252224), (19252224, 23806182)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5295616), (5295616, 9861632), (9861632, 14511104), (14511104, 19175936), (19175936, 22460838)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 3280896), (3280896, 5570048), (5570048, 7826944), (7826944, 9938432), (9938432, 12258304), (12258304, 14592000), (14592000, 16904192), (16904192, 19254272), (19254272, 21574656), (21574656, 23917568), (23917568, 26226176), (26226176, 28570112), (28570112, 30894080), (30894080, 33225216), (33225216, 35551744), (35551744, 37880320), (37880320, 40184832), (40184832, 42524160), (42524160, 44852224), (44852224, 47174144), (47174144, 49514496), (49514496, 51813888), (51813888, 54175744), (54175744, 56489984), (56489984, 58805248), (58805248, 60960000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2],
                side="end",
                delta_sec=-2.00
            )
        elif holes == [(0, 5993472), (5993472, 9967104), (9967104, 14607360), (14607360, 19260416), (19260416, 24106496), (24106496, 28570112), (28570112, 33509376), (33509376, 38098432), (38098432, 42529792), (42529792, 47198720), (47198720, 51844608), (51844608, 56505856), (56505856, 61150720), (61150720, 65804800), (65804800, 70473728), (70473728, 75129856), (75129856, 79778816), (79778816, 84428800), (84428800, 89093632), (89093632, 93743616), (93743616, 98402304), (98402304, 103059968), (103059968, 107782656), (107782656, 112439296), (112439296, 117095424), (117095424, 121913792)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2,3,4],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-2.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_20_batch_03_collet_1_13-02-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")
        if holes == [(0, 9973760), (9973760, 14636544), (14636544, 23037414)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=0.5
            )
            holes = merge_holes_by_index(holes, 2, 3)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 5428736), (5428736, 10085376), (10085376, 14934528), (14934528, 19407360), (19407360, 24040448), (24040448, 28705280), (28705280, 33355264), (33355264, 38139392), (38139392, 42625024), (42625024, 47333376), (47333376, 55671616)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = merge_holes_by_index(holes, 2, 3)
            holes = merge_holes_by_index(holes, 7, 8)
            holes = merge_holes_by_index(holes, 9, 10)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=10,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=-1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=10,
                side="end",
                delta_sec=-2.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,7,8,9,10],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[3,4,5],
                side="end",
                delta_sec=1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5500416), (5500416, 10438656), (10438656, 14795776), (14795776, 19502592), (19502592, 27842214)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=-1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 8635828)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=5.0)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=24.00
            )
        elif holes == [(0, 5801984), (5801984, 13825012)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 4804800)]:
            print("limpo")
        elif holes == [(0, 1527296), (1527296, 3366400), (3366400, 5301760), (5301760, 7631872), (7631872, 11520000)]:
            holes = merge_holes_by_index(holes, 0, 1)
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 2868736), (2868736, 5193216), (5193216, 7520768), (7520768, 9843200), (9843200, 12177920), (12177920, 14498816), (14498816, 16833024), (16833024, 19158528), (19158528, 21490176), (21490176, 23815168), (23815168, 27840000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = merge_holes_by_index(holes, 4, 5)
            holes = merge_holes_by_index(holes, 6, 7)
            holes = merge_holes_by_index(holes, 8, 9)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=10,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 2902528), (2902528, 5224960), (5224960, 7547392), (7547392, 10010624), (10010624, 13920000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 1100800), (1100800, 4320000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=20.00
            )
        elif holes == [(0, 3039232), (3039232, 6912000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 2304000)]:
            print("limpo")
        elif holes == [(0, 3175424), (3175424, 5259264), (5259264, 7610880), (7610880, 9894400), (9894400, 12259328), (12259328, 14587904), (14587904, 17088000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-3.00
            )
        elif holes == [(0, 5357568), (5357568, 10010112), (10010112, 14682624), (14682624, 19505152), (19505152, 23904256), (23904256, 28817408), (28817408, 34171738)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=-1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=-1.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_19_batch_02_collet_1_13-02-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")
        if holes == [(0, 4681728), (4681728, 10160640), (10160640, 14777344), (14777344, 19428864), (19428864, 24232448), (24232448, 28793344), (28793344, 33600512), (33600512, 38060032), (38060032, 42877440), (42877440, 48586138)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 5376512), (5376512, 10067456), (10067456, 14683648), (14683648, 19399168), (19399168, 23976960), (23976960, 28648448), (28648448, 33293312), (33293312, 37959168), (37959168, 42784768), (42784768, 47272448), (47272448, 51930112), (51930112, 56589824), (56589824, 61326336), (61326336, 65879552), (65879552, 70597632), (70597632, 75308032), (75308032, 80018432), (80018432, 84562944), (84562944, 89219584), (89219584, 93897728), (93897728, 98526720), (98526720, 103132672), (103132672, 107839488), (107839488, 112541184), (112541184, 117238272), (117238272, 121901056), (121901056, 126522880), (126522880, 132292160)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5432320), (5432320, 10093056), (10093056, 14756352), (14756352, 19410432), (19410432, 24055808), (24055808, 28704256), (28704256, 33358848), (33358848, 37893120), (37893120, 42681344), (42681344, 47227982)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5588992), (5588992, 9981172)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5497344), (5497344, 10159104), (10159104, 15554740)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 1336320), (1336320, 3646976), (3646976, 5224960), (5224960, 7555072), (7555072, 9882624), (9882624, 12210688), (12210688, 14647808), (14647808, 16988160), (16988160, 19308544), (19308544, 21500928), (21500928, 24288000)]:
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-4.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2,3,4,5,6,7,8],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-1.00
            )
        elif holes == [(0, 1336320), (1336320, 3646976), (3646976, 5224960), (5224960, 7555072), (7555072, 9882624), (9882624, 12210688), (12210688, 14647808), (14647808, 16988160), (16988160, 19308544), (19308544, 21500928), (21500928, 24288000)]:
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-4.00
            )
        elif holes == [(0, 2838528), (2838528, 5185024), (5185024, 7490560), (7490560, 9819648), (9819648, 12140032), (12140032, 14459392), (14459392, 16802304), (16802304, 19130368), (19130368, 21556736), (21556736, 23773696), (23773696, 26257408), (26257408, 28579840), (28579840, 30769152), (30769152, 33197056), (33197056, 35429888), (35429888, 37901312), (37901312, 40211968), (40211968, 42408448), (42408448, 44833280), (44833280, 47057408), (47057408, 49521152), (49521152, 51718656), (51718656, 54045184), (54045184, 56371712), (56371712, 58638336), (58638336, 61150720), (61150720, 63336448), (63336448, 66240000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 2918912), (2918912, 5366784), (5366784, 7872000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=2.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_18_batch_01_collet_1_07-02-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")
        if holes == [(0, 10101760), (10101760, 15939124)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5670912), (5670912, 10211840), (10211840, 14902272), (14902272, 19569664), (19569664, 24226304), (24226304, 33402368), (33402368, 38021632), (38021632, 42881024), (42881024, 47502848), (47502848, 52224512), (52224512, 56850432), (56850432, 61566976), (61566976, 66159104), (66159104, 71038976), (71038976, 75494400), (75494400, 80170496), (80170496, 84826624), (84826624, 89600000), (89600000, 94135296), (94135296, 98751488), (98751488, 103443968), (103443968, 108060160), (108060160, 112735744), (112735744, 117372416), (117372416, 122047488), (122047488, 126681088), (126681088, 131359744), (131359744, 136022016), (136022016, 141876136)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.7
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=-1.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5531136), (5531136, 10008576), (10008576, 14633472), (14633472, 18470400), (18470400, 23454720), (23454720, 27828736), (27828736, 33210778)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=7.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=6.00
            )
        elif holes == [(0, 9788980)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=5.0)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=29.00
            )
        elif holes == [(0, 5477888), (5477888, 10365556)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5484032), (5484032, 10942132)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 5376000), (5376000, 10135552), (10135552, 14017204)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1],
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 5564416), (5564416, 10233856), (10233856, 14885376), (14885376, 19520512), (19520512, 24199680), (24199680, 28850176), (28850176, 33505792), (33505792, 38424576), (38424576, 42813952), (42813952, 47715840), (47715840, 52122624), (52122624, 57977920)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5405184), (5405184, 10097664), (10097664, 14721024), (14721024, 19372032), (19372032, 24015872), (24015872, 28688896), (28688896, 33787354)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 6423040), (6423040, 11085824), (11085824, 15607808), (15607808, 20261376), (20261376, 25006080), (25006080, 29666816), (29666816, 34326528), (34326528, 38993920), (38993920, 43629056), (43629056, 48184320), (48184320, 52841984), (52841984, 57539584), (57539584, 62284288), (62284288, 67008000), (67008000, 71580672), (71580672, 77568692)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5186048), (5186048, 7968000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 1257984), (1257984, 4114432), (4114432, 5745664), (5745664, 8304640), (8304640, 10057728), (10057728, 12385792), (12385792, 14833664), (14833664, 17027072), (17027072, 19161600), (19161600, 21818880), (21818880, 24581632), (24581632, 26247680), (26247680, 28684800), (28684800, 31000576), (31000576, 33328128), (33328128, 35672576), (35672576, 37991936), (37991936, 40325120), (40325120, 42647040), (42647040, 44989440), (44989440, 47310848), (47310848, 49627648), (49627648, 51962880), (51962880, 54626304), (54626304, 56621056), (56621056, 58952704), (58952704, 61274112), (61274112, 63575040), (63575040, 65928704), (65928704, 68224000), (68224000, 70848000)]:
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-9.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="start",
                delta_sec=-2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=-4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=-1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=9,
                side="end",
                delta_sec=-4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=7,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=22,
                side="end",
                delta_sec=-2.00
            )
        elif holes == [(0, 4896000)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=5.0)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=26.00
            )
        elif holes == [(0, 2939904), (2939904, 5382144), (5382144, 7660032), (7660032, 9888256), (9888256, 12239360), (12239360, 14566400), (14566400, 16894464), (16894464, 19222016), (19222016, 21549568), (21549568, 23877120), (23877120, 26205184), (26205184, 28992000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2],
                side="end",
                delta_sec=-1.00
            )
        elif holes == [(0, 2816512), (2816512, 5123072), (5123072, 7443456), (7443456, 9775616), (9775616, 12094464), (12094464, 14429696), (14429696, 16896000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 2808320), (2808320, 5108224), (5108224, 7008000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1],
                side="end",
                delta_sec=2.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_21_batch_02_collet_2_14_02_2025':
        print(holes)
        if holes == [(0, 10056192), (10056192, 14731776), (14731776, 19380736), (19380736, 24117248), (24117248, 28694016), (28694016, 33384960), (33384960, 37922304), (37922304, 42703872), (42703872, 47232512), (47232512, 51881472), (51881472, 60104846)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 4, 5)
            holes = merge_holes_by_index(holes, 6, 7)
            holes = merge_holes_by_index(holes, 8, 9)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=12,
                side="start",
                delta_sec=5.00
            )
        elif holes == [(0, 5529088), (5529088, 10106368), (10106368, 14836224), (14836224, 27934208), (27934208, 33515008), (33515008, 38114816), (38114816, 43576332)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=-8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=-18.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=-9.00
            )
        elif holes == [(0, 9212404)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=5.0)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=26.00
            )
        elif holes == [(0, 5408256), (5408256, 13632820)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="start",
                delta_sec=5.00
            )
        elif holes == [(0, 3588608), (3588608, 5863424), (5863424, 7916544), (7916544, 10231296), (10231296, 12276736), (12276736, 14620160), (14620160, 17166848), (17166848, 19264512), (19264512, 21598720), (21598720, 23923712), (23923712, 26217472), (26217472, 28584448), (28584448, 30048000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=-5.00
            )
        elif holes == [(0, 2875904), (2875904, 5210112), (5210112, 8184320), (8184320, 9883136), (9883136, 14121984), (14121984, 16847872), (16847872, 19167744), (19167744, 21792000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            
            holes = merge_holes_by_index(holes, 0, 1)
            holes = merge_holes_by_index(holes, 2, 3)
            holes = merge_holes_by_index(holes, 5, 6)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="start",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=-6.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=-10.00
            )
        elif holes == [(0, 4512000)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=5.0)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=26.00
            )
        elif holes == [(0, 1574912), (1574912, 3525632), (3525632, 5184000)]:
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-4.00
            )
        elif holes == [(0, 3315712), (3315712, 5324800), (5324800, 6720000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-4.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_16_batch_01_collet_1_06-02-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")
        if holes == [(0, 10137600), (10137600, 14724096), (14724096, 19381248), (19381248, 24035328), (24035328, 28689408), (28689408, 34363930)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 3, 4)
            holes = merge_holes_by_index(holes, 6, 7)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2,3,4,5],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5569024), (5569024, 10331648), (10331648, 14882304), (14882304, 19350528), (19350528, 24013824), (24013824, 28948480), (28948480, 34556122)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5633536), (5633536, 10310144), (10310144, 20923304)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1],
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 5905920), (5905920, 10296832), (10296832, 14939136), (14939136, 23434612)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=6.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=-1.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2],
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 3697152), (3697152, 5289472), (5289472, 6036480), (6036480, 7636480), (7636480, 8348672), (8348672, 9952256), (9952256, 10636288), (10636288, 12277248), (12277248, 14617088), (14617088, 17184000)]:
            holes = merge_holes_by_index(holes, 2, 3)
            holes = merge_holes_by_index(holes, 3, 4)
            holes = merge_holes_by_index(holes, 4, 5)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="start",
                delta_sec=-6.00
            )
        elif holes == [(0, 2951680), (2951680, 5403136), (5403136, 7582720), (7582720, 9917440), (9917440, 12238848), (12238848, 14563328), (14563328, 17034240), (17034240, 19217408), (19217408, 21702656), (21702656, 23878656), (23878656, 26348544), (26348544, 28896000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 1361408), (1361408, 2969088), (2969088, 5293568), (5293568, 7618048), (7618048, 9953280), (9953280, 12286976), (12286976, 14608384), (14608384, 17184000)]:
            holes = merge_holes_by_index(holes, 0, 1)
        elif holes == [(0, 2838528), (2838528, 7008000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 3081728), (3081728, 5282816), (5282816, 7584768), (7584768, 11712000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5424128), (5424128, 10344960), (10344960, 14740480), (14740480, 19408896), (19408896, 24051200), (24051200, 28711424), (28711424, 33363456), (33363456, 38031872), (38031872, 42678272), (42678272, 47328256), (47328256, 51982848), (51982848, 57798542)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10],
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 1500672), (1500672, 2592000)]:
            holes = merge_holes_by_index(holes, 0, 1)
        elif holes == [(0, 5506048), (5506048, 14030016)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_13_batch_01_collet_1_05-02-2025':
        if holes == [(0, 5481984), (5481984, 10098176), (10098176, 14773760), (14773760, 19533824), (19533824, 24103424), (24103424, 28854784), (28854784, 33432576), (33432576, 38068224), (38068224, 42719232), (42719232, 47377920), (47377920, 52204032), (52204032, 56671232), (56671232, 61261824), (61261824, 71610740)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )

        if holes == [(0, 5472000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=0.1
            )
            holes = refine_long_holes_sensive(
                y_ref, sr_ref, holes,
                duration_factor=1.0,
                aggressive_factor=1.5
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=15.00
            )

        if holes == [(0, 1431552), (1431552, 3755520), (3755520, 5252608), (5252608, 7584256), (7584256, 9832960), (9832960, 12160512), (12160512, 14445056), (14445056, 16803840), (16803840, 19241984), (19241984, 21561344), (21561344, 23787008), (23787008, 26114560), (26114560, 28423168), (28423168, 30770176), (30770176, 35808000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="start",
                delta_sec=-6.00
            )

        if holes == [(0, 10942132)]:
            holes = refine_long_holes_sensive(
                y_ref, sr_ref, holes,
                duration_factor=1.0,
                aggressive_factor=1.5
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
        if holes == [(0, 9212404)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=0.1
            )
            holes = refine_long_holes_sensive(
                y_ref, sr_ref, holes,
                duration_factor=1.0,
                aggressive_factor=1.5
            )
        if holes == [(0, 1344000), (1344000, 4608000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=17.00
            )
    if drill.lower() == 'drill_4mm_11_batch_00_collet_1_05-02-2025':
        print(15*"-")
        print(holes)
        print(15*"-")
        if holes == [(0, 10510336), (10510336, 14959616), (14959616, 19603456), (19603456, 24355840), (24355840, 29022720), (29022720, 33579520), (33579520, 38353920), (38353920, 43132416), (43132416, 47639040), (47639040, 52294656), (52294656, 56950784), (56950784, 61593088), (61593088, 66149888), (66149888, 70940160), (70940160, 76607732)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=2.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5261824), (5261824, 14017204)] or holes == [(0, 2906624), (2906624, 7008000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=5.00
            )
        else:
            pass

    if drill.lower() == 'drill_4mm_12_batch_01_collet_1_05-02-2025':
        if holes == [(0, 15930368), (15930368, 22124032), (22124032, 24016896), (24016896, 28717568), (28717568, 33323008), (33323008, 38022656), (38022656, 42737152), (42737152, 47281152), (47281152, 51937280), (51937280, 56591360), (56591360, 61260288), (61260288, 65971712), (65971712, 70627840), (70627840, 77568692)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=2.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="start",
                delta_sec=-8.00
            )
        elif holes == [(0, 5682176), (5682176, 7585792), (7585792, 9824768), (9824768, 12140032), (12140032, 14468096), (14468096, 16795648), (16795648, 19248128), (19248128, 21430272), (21430272, 23907328), (23907328, 26082816), (26082816, 28410880), (28410880, 30842368), (30842368, 33174016), (33174016, 35526144), (35526144, 38688000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=2.0
            )
        elif holes == [(0, 740864), (740864, 1190400), (1190400, 5752948)]:
            holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.9)
        elif holes == [(0, 1085440), (1085440, 4036032)]:
            holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.9)
        else:
            pass
    if drill.lower() == 'drill_4mm_02_batch_00_collet_1_29-01-2025':
        print(15*"-")
        print(holes)
        print(15*"-")
        if holes == [(0, 1353728), (1353728, 2537984), (2537984, 3674624), (3674624, 4748288), (4748288, 6986240), (6986240, 8204288), (8204288, 10692608), (10692608, 11855872), (11855872, 13020672), (13020672, 13987328), (13987328, 16605390)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )

            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=7,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=11,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=11,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=12,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=12,
                side="start",
                delta_sec=-6.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=8,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=8,
                side="start",
                delta_sec=-6.00
            )
        elif holes == [(0, 2207006)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=32.0)
        elif holes == [(0, 3114496), (3114496, 5411328), (5411328, 7697920), (7697920, 9916416), (9916416, 12368896), (12368896, 14551040), (14551040, 16759808), (16759808, 19317248), (19317248, 21530624), (21530624, 24040448), (24040448, 26072576), (26072576, 28725760), (28725760, 33216000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=10,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 1363456), (1363456, 2601472), (2601472, 3768320), (3768320, 4923392), (4923392, 5838848), (5838848, 7008768), (7008768, 8065536), (8065536, 9260032), (9260032, 10403328), (10403328, 11889664), (11889664, 12818944), (12818944, 14229504), (14229504, 16172956)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=7,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=8,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=10,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=12,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=13,
                side="start",
                delta_sec=-4.00
            )
        elif holes == [(0, 2846720), (2846720, 5193728), (5193728, 7578112), (7578112, 9706496), (9706496, 11841536), (11841536, 14030848), (14030848, 16237568), (16237568, 18750464), (18750464, 21539840), (21539840, 23717376), (23717376, 25741824), (25741824, 28396544), (28396544, 32256000)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=4.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=6.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=7,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=10,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=13,
                side="start",
                delta_sec=4.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_04_batch_00_collet_1_30-01-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")
        if holes == [(0, 10279936), (10279936, 14931968), (14931968, 19604480), (19604480, 24277504), (24277504, 28933120), (28933120, 33589248), (33589248, 38260736), (38260736, 42878976), (42878976, 47540224), (47540224, 52193280), (52193280, 56778240), (56778240, 61639680), (61639680, 66128384), (66128384, 70745088), (70745088, 75659776), (75659776, 80062976), (80062976, 84982272), (84982272, 89353216), (89353216, 94133760), (94133760, 98667008), (98667008, 103466496), (103466496, 106955776), (106955776, 112617472), (112617472, 117406720), (117406720, 122215936), (122215936, 126901760), (126901760, 131494912), (131494912, 135925248), (135925248, 139432960), (139432960, 141876134)]:
            holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.7)
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=2.0
            )
            holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.7)
            holes = merge_holes_by_index(holes, 29, 30)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=22,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5237760), (5237760, 7682048), (7682048, 9893376), (9893376, 12338176), (12338176, 14548480), (14548480, 16949248), (16949248, 19287040), (19287040, 21634048), (21634048, 23977984), (23977984, 26260992), (26260992, 28598784), (28598784, 30961664), (30961664, 33288192), (33288192, 35628544), (35628544, 37838336), (37838336, 40153600), (40153600, 42481152), (42481152, 44784128), (44784128, 47252480), (47252480, 49549312), (49549312, 51926016), (51926016, 54119936), (54119936, 56550400), (56550400, 58775552), (58775552, 61084672), (61084672, 63430656), (63430656, 65758208), (65758208, 68086272), (68086272, 70848000)]:
            holes = refine_long_holes_sensive(
                y_ref, sr_ref, holes,
                duration_factor=1.0,
                aggressive_factor=1.5
            )
            holes = merge_short_holes_by_median(holes, sr_ref, factor=0.5)
            holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.7)

            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=14.00
            )
        elif holes == [(0, 5522432), (5522432, 10142208), (10142208, 14801920), (14801920, 19571200), (19571200, 24112128), (24112128, 29026816), (29026816, 33495040), (33495040, 38078976), (38078976, 44934490)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5475840), (5475840, 10140672), (10140672, 14766080), (14766080, 19336704), (19336704, 24049664), (24049664, 28926976), (28926976, 33396224), (33396224, 37966336), (37966336, 42686976), (42686976, 49905856)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5690368), (5690368, 10326528), (10326528, 14401588)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 2994688), (2994688, 5306368), (5306368, 7629312), (7629312, 9975296), (9975296, 12255744), (12255744, 14384640), (14384640, 16911872), (16911872, 19271680), (19271680, 22464000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[5],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7],
                side="end",
                delta_sec=0.50
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_10_batch_00_collet_1_04-02-2025':
        holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.5)
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")

        if holes == [(0, 3126784), (3126784, 5454848), (5454848, 7780352), (7780352, 10125312), (10125312, 11924992), (11924992, 14720000), (14720000, 17084416), (17084416, 19296000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6],
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 5423104), (5423104, 10173364)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 6390272), (6390272, 11016704), (11016704, 15409152), (15409152, 20255232), (20255232, 24395572)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1],
                side="end",
                delta_sec=-1.00
            )
        elif holes == [(0, 5516800), (5516800, 10176000), (10176000, 14845952), (14845952, 19469824), (19469824, 24137728), (24137728, 28801024), (28801024, 33457664), (33457664, 38211584), (38211584, 42767360), (42767360, 47537664), (47537664, 52261888), (52261888, 57414156)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5805568), (5805568, 10373120), (10373120, 15149056), (15149056, 19771392), (19771392, 24450560), (24450560, 29169152), (29169152, 33735680), (33735680, 38604966)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 2988032), (2988032, 6432000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 5722624), (5722624, 13056244)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 2875392), (2875392, 5334528), (5334528, 7527424), (7527424, 9846784), (9846784, 12185088), (12185088, 14517248), (14517248, 16821760), (16821760, 19082752), (19082752, 21499904), (21499904, 23749120), (23749120, 26153472), (26153472, 28608000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1],
                side="end",
                delta_sec=-1.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[7],
                side="end",
                delta_sec=1.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[8],
                side="end",
                delta_sec=0.50
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[9],
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 3316736), (3316736, 5652480), (5652480, 7936000), (7936000, 10266624), (10266624, 12096000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3],
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 2790912), (2790912, 5117952), (5117952, 7444992), (7444992, 9772544), (9772544, 12108288), (12108288, 14427648), (14427648, 16747520), (16747520, 19221504), (19221504, 21549568), (21549568, 23846912), (23846912, 26064896), (26064896, 28531712), (28531712, 30829056), (30829056, 33146880), (33146880, 35491328), (35491328, 37813248), (37813248, 40128000), (40128000, 42357248), (42357248, 44813824), (44813824, 47123968), (47123968, 49450496), (49450496, 51769344), (51769344, 54002688), (54002688, 57504000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5486592), (5486592, 10170368), (10170368, 14791680), (14791680, 19312128), (19312128, 24082944), (24082944, 28621312), (28621312, 33280000), (33280000, 38057472), (38057472, 42605568), (42605568, 47557632), (47557632, 51907584), (51907584, 56677888), (56677888, 61532672), (61532672, 65864704), (65864704, 70523392), (70523392, 75196416), (75196416, 79832576), (79832576, 84496384), (84496384, 89144320), (89144320, 93794304), (93794304, 98462208), (98462208, 103113728), (103113728, 107854848), (107854848, 115007694)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 6374912), (6374912, 10131456), (10131456, 14618624), (14618624, 19405312), (19405312, 24016384), (24016384, 28575232), (28575232, 34748314)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 3367936), (3367936, 5249024), (5249024, 7575552), (7575552, 9914880), (9914880, 12226560), (12226560, 16280576), (16280576, 17280000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-5.50
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=-18.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[1,2,3,4],
                side="end",
                delta_sec=1.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_09_batch_00_collet_1_31-01-2025':

        holes = refine_long_holes_sensive(
            y_ref, sr_ref, holes,
            duration_factor=1.0,
            aggressive_factor=1.5
        )
        holes = merge_short_holes_by_median(holes, sr_ref, factor=0.4)
        holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.5)

        print(15*"-")
        print(holes)
        print(15*"-")

        if holes == [(0, 2849792), (2849792, 4896000)] or holes == [(0, 5502464), (5502464, 9801792)]:
            # holes = merge_holes_by_index(holes, -1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 3124736), (3124736, 5453312), (5453312, 7785984), (7785984, 10111488), (10111488, 14112000)] or holes == [(0, 5727744), (5727744, 10384384), (10384384, 15038464), (15038464, 19681792), (19681792, 28213786)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.5
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 5075968), (5075968, 10186752), (10186752, 14827520), (14827520, 19499520), (19499520, 24150528), (24150528, 28800000), (28800000, 33443328), (33443328, 38106624), (38106624, 42771968), (42771968, 47427072), (47427072, 52082688), (52082688, 56738304), (56738304, 61385728), (61385728, 66052096), (66052096, 70706688), (70706688, 75354112), (75354112, 80006144), (80006144, 84662784), (84662784, 89316352), (89316352, 93978624), (93978624, 98637312), (98637312, 103465472), (103465472, 107945984), (107945984, 112589312), (112589312, 117243392), (117243392, 121899520), (121899520, 126550016), (126550016, 131212288), (131212288, 135878656), (135878656, 141696756)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
                side="end",
                delta_sec=2.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 6171136), (6171136, 10768896), (10768896, 15388160), (15388160, 20028416), (20028416, 24682496), (24682496, 29319680), (29319680, 34070528), (34070528, 38706688), (38706688, 43395072), (43395072, 48012800), (48012800, 52707328), (52707328, 57321472), (57321472, 61922816), (61922816, 67971904)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 2846208), (2846208, 5209600), (5209600, 7537664), (7537664, 9865216), (9865216, 12168704), (12168704, 14522368), (14522368, 16850432), (16850432, 19177984), (19177984, 21477888), (21477888, 23799808), (23799808, 26140160), (26140160, 28459520), (28459520, 30783488), (30783488, 33107968), (33107968, 35449344), (35449344, 37776896), (37776896, 40086528), (40086528, 42455040), (42455040, 44783104), (44783104, 47110656), (47110656, 49388544), (49388544, 51720192), (51720192, 54093824), (54093824, 56419328), (56419328, 58721280), (58721280, 61167616), (61167616, 63493632), (63493632, 65854464), (65854464, 68060160), (68060160, 70848000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
                side="end",
                delta_sec=2.00
            )
        else:
            pass

    if drill.lower() == 'drill_4mm_05_batch_00_collet_1_30-01-2025':
        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")

        if holes == [(0, 2918400), (2918400, 5268480), (5268480, 7578112), (7578112, 9906176), (9906176, 12163072), (12163072, 14541312), (14541312, 16722944), (16722944, 19236352), (19236352, 21556224), (21556224, 23879680), (23879680, 26145792), (26145792, 28430336), (28430336, 30851072), (30851072, 33122304), (33122304, 35475456), (35475456, 37833216), (37833216, 40106496), (40106496, 42501120), (42501120, 44784128), (44784128, 47155200), (47155200, 49491456), (49491456, 51811328), (51811328, 54042112), (54042112, 56461824), (56461824, 58806784), (58806784, 61127680), (61127680, 63432192), (63432192, 65772032), (65772032, 68117504), (68117504, 70752000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-1.50
            )
        elif holes == [(0, 3533312), (3533312, 5327360), (5327360, 7631360), (7631360, 9960960), (9960960, 12289536), (12289536, 14596608), (14596608, 16943104), (16943104, 19272704), (19272704, 21580288), (21580288, 23909376), (23909376, 26331136), (26331136, 28584960), (28584960, 30892544), (30892544, 33348096), (33348096, 35674624), (35674624, 37871104), (37871104, 40194560), (40194560, 42528256), (42528256, 44878336), (44878336, 47330304), (47330304, 49515008), (49515008, 51841536), (51841536, 54240256), (54240256, 56487424), (56487424, 58848256), (58848256, 61156864), (61156864, 63503872), (63503872, 65831936), (65831936, 68240384), (68240384, 70944000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=1.00
            )
        elif holes == [(0, 5340672), (5340672, 10059264), (10059264, 14618112), (14618112, 19253760), (19253760, 24006656), (24006656, 28585984), (28585984, 33488896), (33488896, 37865472), (37865472, 42706944), (42706944, 47518720), (47518720, 51866624), (51866624, 56516608), (56516608, 61213696), (61213696, 65911808), (65911808, 70530560), (70530560, 75190272), (75190272, 79778816), (79778816, 84456960), (84456960, 89145856), (89145856, 93767680), (93767680, 98486784), (98486784, 103188480), (103188480, 107815424), (107815424, 112595456), (112595456, 116879872), (116879872, 121813504), (121813504, 126466048), (126466048, 131089408), (131089408, 135743488), (135743488, 141888948)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5459968), (5459968, 10110976), (10110976, 15170356)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5649920), (5649920, 10185216), (10185216, 14838784), (14838784, 19335168), (19335168, 24239104), (24239104, 28859904), (28859904, 33306624), (33306624, 37965824), (37965824, 42809856), (42809856, 47532032), (47532032, 52200448), (52200448, 56751616), (56751616, 61347840), (61347840, 65894912), (65894912, 70719488), (70719488, 75340800), (75340800, 80034304), (80034304, 84743680), (84743680, 89282048), (89282048, 94111232), (94111232, 98622976), (98622976, 103330816), (103330816, 107955200), (107955200, 112450048), (112450048, 117092864), (117092864, 121991168), (121991168, 126530048), (126530048, 131352576), (131352576, 135902208), (135902208, 141696756)] or holes == [(0, 3533312), (3533312, 5327360), (5327360, 7631360), (7631360, 9960960), (9960960, 12289536), (12289536, 14596608), (14596608, 16943104), (16943104, 19272704), (19272704, 21580288), (21580288, 23909376), (23909376, 26331136), (26331136, 28584960), (28584960, 30892544), (30892544, 33348096), (33348096, 35674624), (35674624, 37871104), (37871104, 40194560), (40194560, 42528256), (42528256, 44878336), (44878336, 47330304), (47330304, 49515008), (49515008, 51841536), (51841536, 54240256), (54240256, 56487424), (56487424, 58848256), (58848256, 61156864), (61156864, 63503872), (63503872, 65831936), (65831936, 68240384), (68240384, 70944000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-1.00
            )
        elif holes == [(0, 2801152), (2801152, 5130240), (5130240, 7447040), (7447040, 9780224), (9780224, 12111360), (12111360, 14411776), (14411776, 16747008), (16747008, 19011584), (19011584, 21418496), (21418496, 23747584), (23747584, 26496000)]:
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,4,5,6,7,8,9],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 5404160), (5404160, 10070528), (10070528, 14638592), (14638592, 19466752), (19466752, 22496768), (22496768, 27969024), (27969024, 31693312), (31693312, 36643840), (36643840, 41065472), (41065472, 47315456), (47315456, 52993742)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=9.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=5,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=6,
                side="end",
                delta_sec=9.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=7,
                side="end",
                delta_sec=8.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=8,
                side="end",
                delta_sec=9.00
            )
            holes = adjust_hole_boundaries_by_ids(
                holes, sr_ref,
                hole_ids=[0,1,2,3,9],
                side="end",
                delta_sec=2.00
            )
        elif holes == [(0, 2882560), (2882560, 5187584), (5187584, 7584000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 3151872), (3151872, 7200000)] or holes == [(0, 5781504), (5781504, 14593780)]:
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=3.00
            )
            if holes == [(0, 3151872), (3151872, 7200000)]:
                holes = adjust_hole_boundary(
                    holes, sr_ref,
                    hole_idx=1,
                    side="end",
                    delta_sec=4.00
                )
        else:
            pass

    if drill.lower() == 'drill_4mm_07_batch_00_collet_1_31-01-2025':

        if holes == [(0, 10060800), (10060800, 14719488), (14719488, 23229606)]:

            holes = refine_long_holes_sensive(
                y_ref, sr_ref, holes,
                duration_factor=0.1,
                aggressive_factor=0.3
            )

            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=4.00
            )
        elif holes == [(0, 1278464), (1278464, 2963456), (2963456, 5291008), (5291008, 7596032), (7596032, 8664064), (8664064, 11616000)] or \
                holes == [(0, 5536256), (5536256, 10407936), (10407936, 14745600), (14745600, 19412992), (19412992, 24225792), (24225792, 28845056), (28845056, 33466880), (33466880, 38349824), (38349824, 42690048), (42690048, 47348736), (47348736, 52240896), (52240896, 56655360), (56655360, 61316096), (61316096, 65962496), (65962496, 70615552), (70615552, 75274240), (75274240, 79934976), (79934976, 84897280), (84897280, 89521152), (89521152, 93897216), (93897216, 98545664), (98545664, 103216640), (103216640, 107855360), (107855360, 112511488), (112511488, 117170688), (117170688, 125962638)] or \
                holes == [(0, 2837504), (2837504, 5277184), (5277184, 7467008), (7467008, 9942528), (9942528, 12227584), (12227584, 14610432), (14610432, 16925184), (16925184, 19236352), (19236352, 21421568), (21421568, 23905792), (23905792, 26090496), (26090496, 28555776), (28555776, 30877184), (30877184, 33187328), (33187328, 35550208), (35550208, 37877248), (37877248, 40170496), (40170496, 42529792), (42529792, 44872704), (44872704, 47205888), (47205888, 49467904), (49467904, 51698176), (51698176, 54186496), (54186496, 56352768), (56352768, 58811904), (58811904, 62880000)]:
            if holes == [(0, 1278464), (1278464, 2963456), (2963456, 5291008), (5291008, 7596032), (7596032, 8664064), (8664064, 11616000)]:
                holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.6)
            else:
                holes = refine_long_holes_sensive(
                    y_ref, sr_ref, holes,
                    duration_factor=0.5,
                    aggressive_factor=1.0
                )
                holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.6)
        else:
            holes = merge_short_holes_by_median(holes, sr_ref, factor=0.6)
            holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.6)
    if drill.lower() == 'drill_4mm_06_batch_00_collet_1_30-01-2025':
        print(holes)
        holes = merge_short_holes_by_median(holes, sr_ref, factor=0.6)
        holes = refine_long_holes_sensive(
            y_ref, sr_ref, holes,
            duration_factor=1.5,
            aggressive_factor=2.0
        )
        holes = expand_holes_backward(holes)
        holes = merge_short_holes_by_recursion(holes, sr_ref, factor=0.6)
        if holes == [(0, 3209728), (3209728, 7968000)] or holes == [(0, 6019584), (6019584, 15926312)]:
            print( "Aplicando split forçado para este caso específico")
            print( "Aplicando split forçado para este caso específico")
            print( "Aplicando split forçado para este caso específico")
            print( "Aplicando split forçado para este caso específico")
            holes = force_temporal_split_holes(
                holes,
                sr_ref,
                duration_factor=1.0
            )
    if drill.lower() == 'drill_4mm_15_batch_01_collet_1_06-02-2025':

        holes = refine_long_holes_sensive(
            y_ref, sr_ref, holes,
            duration_factor=0.5,
            aggressive_factor=1.0
        )
        holes = merge_short_holes_by_median(holes, sr_ref, factor=0.5)
        holes = merge_short_holes_by_median(holes, sr_ref, factor=0.6)

        print(15*"-")
        print(take_path)
        print(holes)
        print(15*"-")

        if holes == [(0, 4800000)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=34.0)
        elif holes == [(0, 9404596)]:
            holes = add_initial_hole_and_reduce(holes, sr_ref, duration_sec=32.0)
        elif holes == [(0, 2768896), (2768896, 5933056), (5933056, 7583232), (7583232, 9437696), (9437696, 10752000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=-5.00
            )
            holes = merge_holes_by_index(holes, 3, 4)
        elif holes == [(0, 2883584), (2883584, 5216256), (5216256, 7872000)] or holes == [(0, 5541376), (5541376, 10188800), (10188800, 15746932)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 2808832), (2808832, 5301248), (5301248, 7465472), (7465472, 9696000)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=1.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 4786176), (4797440, 10185216), (10185216, 14896128), (14896128, 21692070)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=5.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 6776832), (6776832, 11424768), (11424768, 16073728), (16073728, 20733952), (20733952, 26291866)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 5605888), (5605888, 10254848), (10254848, 15218176), (15218176, 19665408), (19665408, 24292352), (24292352, 29751322)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=4,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 5623296), (5623296, 10147840), (10147840, 14703616), (14703616, 19398580)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 5523968), (5523968, 9981172)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 2889728), (2889728, 5280000)] or holes == [(0, 5553152), (5553152, 10749940)]:
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=3.00
            )
        elif holes == [(0, 2075648), (2075648, 4625920), (4625920, 6460416), (6460416, 8947200), (8947200, 11465216), (11465216, 13152000)]:
            holes = merge_holes_by_index(holes, 0, 1)
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=0,
                side="end",
                delta_sec=-10.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=1,
                side="end",
                delta_sec=-3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=2,
                side="end",
                delta_sec=-3.00
            )
            holes = adjust_hole_boundary(
                holes, sr_ref,
                hole_idx=3,
                side="end",
                delta_sec=-5.00
            )
        else:
            pass
    if drill.lower() == 'drill_4mm_14_batch_01_collet_1_06-02-2025':
        holes = merge_short_holes_by_median(holes, sr_ref, factor=0.7)

        holes = refine_long_holes_sensive(
            y_ref, sr_ref, holes,
            duration_factor=0.5,
            aggressive_factor=1.0
        )
        holes = merge_short_holes_by_median(holes, sr_ref, factor=0.6)
    if drill.lower() == 'drill_4mm_08_batch_00_collet_1_31-01-2025':
        holes = merge_short_holes_by_median(holes, sr_ref, factor=0.7)


    holes_ref = holes
    N = len(holes_ref)

    print(f"TAKE {os.path.basename(take_path)}: {N}")

    rel_path = os.path.relpath(take_path, standardized_root)
    take_dir = os.path.join(segmented_root, rel_path)
    ensure_dir(take_dir)

    img_dir = os.path.join(DOCS_IMG_DIR, drill, os.path.basename(take_path))
    ensure_dir(img_dir)

    plot_spectrogram_with_holes(
        y_ref, sr_ref, holes_ref,
        os.path.join(img_dir, f"{os.path.basename(take_path)}_spectrogram.png")
    )

    jam_problem_flag = len(jams_list) > 0 and N < len(jams_list)
    overview_records = []

    column_hole_id = hole_counter["column_id"]

    take_hole_index = 0 + column_hole_id

    for idx, file in enumerate(wavs):
        file_path = os.path.join(take_path, file)
        filebase = os.path.splitext(file)[0]

        fname = file.lower()

        is_ultra = ("ultrasonic" in fname) or ("ult" in fname) or ("ts" in fname)
        mic_type = "ultra" if is_ultra else "common"

        position = (
            "internal" if "_int" in fname
            else "external" if "_ext" in fname
            else "unknown"
        )

        try:
            if is_ultra:
                y_full, sr = sf.read(file_path, always_2d=True)
                y_mono = np.mean(y_full, axis=1)
            else:
                y_mono, sr = librosa.load(file_path, sr=sr_ref, mono=True)
        except Exception as ex:
            print(f"Failed to load {file_path}: {ex}")
            continue

        holes_this_mic = align_holes_to_reference(
            holes_ref,
            y_mono,
            sr,
            tolerance_sec=ALIGN_TOLERANCE_SEC
        )

        if len(holes_this_mic) != N:
            holes_this_mic = holes_ref.copy()

        for local_idx, (s, e) in enumerate(holes_this_mic, start=1):
            hole_num = local_idx + column_hole_id -1

            jam = hole_num in jams_list
            suffix = "_jam" if jam else ""

            out_name = f"{filebase}_hole{hole_num:05d}{suffix}.wav"
            out_path = os.path.join(take_dir, out_name)

            sf.write(out_path, y_mono[int(s):int(e)], sr)

            voltage_list, current_list = extract_datalogger_window(
                datalog_df,
                hole_start_sec=s / sr,
                hole_end_sec=e / sr,
                audio_start_time=audio_start_time
            )

            holes_before_fail = compute_holes_before_fail(
                hole_num,
                jams_list
            )

            overview_records.append(dict(
                file=file,
                hole_id=hole_num,
                local_hole_idx=local_idx,
                start_sec=s / sr,
                end_sec=e / sr,
                duration_sec=(e - s) / sr,
                jam_flag=jam,
                jam_problem=jam_problem_flag,
                holes_before_fail=holes_before_fail,
                mic_type=mic_type,
                position=position,
                mic_filename=file,
                output_path=out_path,
                filepath_features="",
                Voltage=voltage_list,
                Current=current_list
            ))

            take_hole_index += 1

    df = pd.DataFrame(overview_records)
    if not df.empty:
        metadata.append(df)

    num_holes = local_idx 
    return num_holes

def extract_column_key(path):
    import os

    normalized = os.path.normpath(path)

    print("DEBUG normalized:", normalized)

    for part in normalized.split(os.sep):
        print("DEBUG part:", part)

        if part.lower().startswith("column_"):
            print("DEBUG column key found:", part)
            return part

    print("DEBUG no column found")
    return None

if __name__ == "__main__":
    print("Starting script")

    ensure_dir(METADATA_DIR)
    ensure_dir(SEGMENTED_DIR)
    ensure_dir(DOCS_IMG_DIR)

    metadata_all = []
    drills = ['drill_4mm_10_batch_00_collet_1_04-02-2025']

    print(f"Starting drill processing in {STANDARDIZED_DIR}...")

    for drill in drills:
        drill_path = os.path.join(STANDARDIZED_DIR, drill)
        jams_list = load_jams_for_drill(drill_path)

        global_hole_offset = 0

        current_column_key = None
        current_take_root = None

        column_holes_count = 0
        column_start_id = 1

        hole_counter = {"column_id": 1}

        roots_to_process = []

        for root, dirs, files in os.walk(drill_path):

            dirs[:] = [
                d for d in dirs
                if "collect_test" not in d.lower()
                   and "collet_test" not in d.lower()
                   and "collet_test_2" not in d.lower()
            ]

            dirs.sort(key=column_folder_key)

            if any(f.lower().endswith(".wav") for f in files):
                roots_to_process.append(root)

        roots_to_process.sort(key=column_folder_key)

        for root in roots_to_process:

            sensor_parent = os.path.dirname(root)
            sensor_type = os.path.basename(sensor_parent).lower()

            take_name = os.path.basename(root)

            take_root = os.path.join(
                os.path.dirname(sensor_parent),
                take_name
            )

            column_key = extract_column_key(take_root)

            print("root:", root)
            print("sensor_type:", sensor_type)
            print("take_root:", take_root)
            print("column_key:", column_key)
            print("current_column_key:", current_column_key)
            print("global_hole_offset:", global_hole_offset)
            print("column_holes_count:", column_holes_count)
            print("column_start_id:", column_start_id)
            print("hole_counter antes:", hole_counter)

            if column_key != current_column_key:
                if current_column_key is not None:
                    global_hole_offset += column_holes_count

                    print(
                        f"Finishing {current_column_key}: "
                        f"{column_holes_count} holes"
                    )
                    print(
                        f"global_hole_offset updated to "
                        f"{global_hole_offset}"
                    )

                current_column_key = column_key
                current_take_root = take_root

                column_holes_count = 0

                column_start_id = global_hole_offset + 1

                hole_counter = {
                    "column_id": column_start_id
                }

                print(f"New column detected: {column_key}")
                print(
                    f"hole_counter started at "
                    f"{hole_counter['column_id']}"
                )

            elif take_root != current_take_root:

                current_take_root = take_root

                print(
                    f"New TAKE detected in the same column: "
                    f"{take_root}"
                )

                hole_counter = {
                    "column_id": column_start_id
                }

                print(
                    f"hole_counter reset to "
                    f"{hole_counter['column_id']}"
                )

            print(f"Processing {root}")

            try:
                holes_in_take = process_take_folder(
                    root,
                    drill,
                    jams_list,
                    metadata_all,
                    hole_counter
                )

            except Exception as e:
                print(f"Error processing {root}: {e}")
                holes_in_take = 0

            print(
                f"After processing {root}: "
                f"hole_counter={hole_counter}, "
                f"holes_in_take={holes_in_take}, "
                f"sensor_type={sensor_type}"
            )

            if holes_in_take and sensor_type == "ultrasonic_mics":

                column_holes_count = holes_in_take

                hole_counter["column_id"] = column_start_id + holes_in_take

                print(
                    f"{hole_counter['column_id']} | "
                    f"column_holes_count={column_holes_count}"
                )

            elif holes_in_take:

                print(
                    "ℹ reg_mics detected: counter not incremented"
                )

        if current_column_key is not None:
            global_hole_offset += column_holes_count

            print(
                f"Finishing last column {current_column_key}: "
                f"{column_holes_count} holes"
            )
            print(
                f"Final accumulated total = {global_hole_offset}"
            )

        print(
            f"Drill finished: {drill} | "
            f"Total accumulated holes = {global_hole_offset}"
        )

    if metadata_all:

        metadata_valid = [
            df for df in metadata_all
            if df is not None and not df.empty
        ]

        if metadata_valid:
            df_all = pd.concat(metadata_valid, ignore_index=True)
            df_all.to_csv(METADATA_CSV, index=False)

            print(f"Complete metadata saved to {METADATA_CSV}")

        else:
            print("No valid metadata to save.")

    print("Processing complete.")