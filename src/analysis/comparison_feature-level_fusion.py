import gc
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping

from src.models.hybrid_cnn_2d import build_late_fusion_architecture


os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.get_logger().setLevel("ERROR")

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(
    os.path.join(SCRIPT_PATH, "..", "..", "data", "cache")
)
RESULTS_DIR = os.path.abspath(
    os.path.join(SCRIPT_PATH, "..", "..", "results")
)
HOLE_SCORES_DIR = os.path.join(RESULTS_DIR, "feature_fusion_hole_scores")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(HOLE_SCORES_DIR, exist_ok=True)

CACHE_FILE = os.path.join(CACHE_DIR, "processed_dataset.npz")
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(f"NPZ file not found: {CACHE_FILE}")

loaded = np.load(CACHE_FILE, allow_pickle=True)
drills_data = loaded["drills_data"].item()
loaded.close()

if not drills_data:
    raise RuntimeError("drills_data is empty.")

ALL_DRILLS = sorted(list(drills_data.keys()))
DOMAINS = ["regular", "ultrasonic"]
MIC_CONFIGS = ["1mic", "3mic"]

ONE_MIC_EXTERNAL = [1]
ONE_MIC_INTERNAL = [4]
THREE_MIC_EXTERNAL = [1, 2, 3]
THREE_MIC_INTERNAL = [4, 5, 6]

LEARNING_RATE = 5e-4
EPOCHS = 60
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 10


def reset_random_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def apply_paper_criteria(X, y_jam, holes):
    unique_holes = np.sort(np.unique(holes))
    n_total = len(unique_holes)

    if n_total == 0:
        return (
            np.empty_like(X),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    jam_holes = {int(h) for h in unique_holes if np.any(y_jam[holes == h] == 1)}
    non_jam_holes = [int(h) for h in unique_holes if int(h) not in jam_holes]

    healthy_holes = set(non_jam_holes[:5])

    anomaly_start = int(np.floor(0.80 * n_total))
    anomaly_holes = set(int(h) for h in unique_holes[anomaly_start:]) - jam_holes
    anomaly_holes = anomaly_holes - healthy_holes

    valid_holes = healthy_holes | anomaly_holes
    mask = np.array([int(h) in valid_holes for h in holes], dtype=bool)

    X_filtered = X[mask]
    holes_filtered = holes[mask]
    y_new = np.array(
        [0.0 if int(h) in healthy_holes else 1.0 for h in holes_filtered],
        dtype=np.float32,
    )

    return X_filtered, y_new, holes_filtered


def compute_metrics_custom(y_true, y_pred, scores):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    mcc = (
        matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    )

    try:
        auc = (
            0.5 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, scores)
        )
    except Exception:
        auc = 0.5

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "mcc": mcc,
    }


def get_tr_data(data, tr, position, domain):
    prefix = f"TR{tr}_{position}_{domain}"
    return (
        np.asarray(data[f"X_{prefix}"], dtype=np.float32),
        np.asarray(data[f"y_{prefix}"], dtype=np.float32),
        np.asarray(data[f"holes_{prefix}"], dtype=np.int32),
    )


def align_trs_by_hole(data, trs, position, domain):
    tr_data = {}
    for tr in trs:
        X, y, holes = get_tr_data(data, tr, position, domain)
        hole_to_idx = {int(h): i for i, h in enumerate(holes)}
        tr_data[tr] = {
            "X": X,
            "y": y,
            "holes": holes,
            "hole_to_idx": hole_to_idx,
        }

    common_holes = None
    for tr in trs:
        c_set = set(tr_data[tr]["hole_to_idx"].keys())
        common_holes = c_set if common_holes is None else common_holes & c_set

    common_holes = sorted(common_holes) if common_holes else []
    aligned = {}
    for tr in trs:
        indices = [tr_data[tr]["hole_to_idx"][h] for h in common_holes]
        aligned[tr] = {
            "X": tr_data[tr]["X"][indices],
            "y": tr_data[tr]["y"][indices],
            "holes": np.asarray(common_holes, dtype=np.int32),
        }
    return aligned


def build_position_configuration(data, domain, trs, position):
    aligned = align_trs_by_hole(data, trs, position, domain)
    if len(trs) == 1:
        tr = trs[0]
        return {
            "X": aligned[tr]["X"].astype(np.float32),
            "y": aligned[tr]["y"].astype(np.float32),
            "holes": aligned[tr]["holes"].astype(np.int32),
        }
    else:
        X_fused = np.mean(
            np.stack([aligned[tr]["X"] for tr in trs], axis=0), axis=0
        ).astype(np.float32)
        return {
            "X": X_fused,
            "y": aligned[trs[0]]["y"].astype(np.float32),
            "holes": aligned[trs[0]]["holes"].astype(np.int32),
        }


def get_aligned_domain_data(data, domain, mic_cfg, filter_criteria=True):
    ext_trs = (
        ONE_MIC_EXTERNAL if mic_cfg == "1mic" else THREE_MIC_EXTERNAL
    )
    int_trs = (
        ONE_MIC_INTERNAL if mic_cfg == "1mic" else THREE_MIC_INTERNAL
    )

    internal = build_position_configuration(data, domain, int_trs, "internal")
    external = build_position_configuration(data, domain, ext_trs, "external")

    common_h = np.intersect1d(
        np.unique(internal["holes"]), np.unique(external["holes"])
    )
    idx_int = [
        np.where(internal["holes"] == h)[0][0]
        for h in common_h
        if np.any(internal["holes"] == h)
    ]
    idx_ext = [
        np.where(external["holes"] == h)[0][0]
        for h in common_h
        if np.any(external["holes"] == h)
    ]

    if len(idx_int) != len(common_h) or len(idx_ext) != len(common_h):
        raise RuntimeError(
            f"Internal/External alignment failure | domain={domain} |"
            f" mic_cfg={mic_cfg}"
        )

    if not np.array_equal(
            internal["holes"][idx_int],
            external["holes"][idx_ext],
    ):
        raise RuntimeError(
            f"Misaligned internal/external holes | domain={domain}"
        )

    X_int_raw = internal["X"][idx_int]
    X_ext_raw = external["X"][idx_ext]
    y_raw_int = internal["y"][idx_int]
    y_raw_ext = external["y"][idx_ext]
    holes_raw = internal["holes"][idx_int]

    if not np.array_equal(
            np.asarray(y_raw_int, dtype=np.float32),
            np.asarray(y_raw_ext, dtype=np.float32),
    ):
        raise RuntimeError(
            f"Different JAM labels between Internal and External | domain={domain}"
        )

    if filter_criteria:
        X_int_f, y_f, holes_f = apply_paper_criteria(
            X_int_raw, y_raw_int, holes_raw
        )
        valid_holes = set(holes_f)
        ext_mask = np.array([int(h) in valid_holes for h in holes_raw], dtype=bool)
        X_ext_f = X_ext_raw[ext_mask]
        return X_int_f, X_ext_f, y_f, holes_f
    else:
        return X_int_raw, X_ext_raw, y_raw_int, holes_raw


def audit_test_isolation(all_drills, test_drill):
    all_d = set(str(d) for d in all_drills)
    train_drills = all_d - {str(test_drill)}
    if str(test_drill) in train_drills:
        raise RuntimeError(
            f"LODO LEAK: test drill '{test_drill}' is in the drill string"
        )


def audit_shapes(X_int, X_ext, y, holes, name):
    if not (len(X_int) == len(X_ext) == len(y) == len(holes)):
        raise RuntimeError(
            f"Size inconsistency in {name}: int={len(X_int)},"
            f" ext={len(X_ext)}, y={len(y)}, holes={len(holes)}"
        )
    if X_int.shape[1:] != X_ext.shape[1:]:
        raise RuntimeError(
            f"Incompatible shapes: int={X_int.shape[1:]}, ext={X_ext.shape[1:]}"
        )

all_results = []

for DOMAIN in DOMAINS:
    for MIC_CFG in MIC_CONFIGS:
        print(f"\nDOMAIN: {DOMAIN.upper()} | MICS: {MIC_CFG.upper()}")

        for drill_test in ALL_DRILLS:
            print(f"LODO Drill Test: {drill_test}")
            clear_session()
            gc.collect()
            reset_random_seeds()

            audit_test_isolation(ALL_DRILLS, drill_test)

            train_int, train_ext, train_y, train_holes, train_drills = (
                [],
                [],
                [],
                [],
                [],
            )
            for d_name in ALL_DRILLS:
                if d_name == drill_test:
                    continue
                Xi, Xe, y, h = get_aligned_domain_data(
                    drills_data[d_name], DOMAIN, MIC_CFG, filter_criteria=True
                )
                if len(Xi) > 0:
                    train_int.append(Xi)
                    train_ext.append(Xe)
                    train_y.append(y)
                    train_holes.append(h)
                    train_drills.append(np.full(len(h), d_name, dtype=object))

            X_train_int = np.concatenate(train_int, axis=0)
            X_train_ext = np.concatenate(train_ext, axis=0)
            y_train = np.concatenate(train_y, axis=0)
            holes_train = np.concatenate(train_holes, axis=0)
            drills_train = np.concatenate(train_drills, axis=0)

            X_test_int, X_test_ext, y_test, holes_test = get_aligned_domain_data(
                drills_data[drill_test], DOMAIN, MIC_CFG, filter_criteria=True
            )

            (
                X_all_int,
                X_all_ext,
                y_all_raw,
                holes_all,
            ) = get_aligned_domain_data(
                drills_data[drill_test], DOMAIN, MIC_CFG, filter_criteria=False
            )

            audit_shapes(X_train_int, X_train_ext, y_train, holes_train, "TRAIN")
            audit_shapes(X_test_int, X_test_ext, y_test, holes_test, "TEST")

            train_keys = np.asarray(
                [f"{d}__{h}" for d, h in zip(drills_train, holes_train)], dtype=object
            )
            unique_keys = np.unique(train_keys)
            key_labels = np.asarray(
                [int(np.max(y_train[train_keys == k])) for k in unique_keys],
                dtype=np.int32,
            )

            keys_tr_split, keys_val_split = train_test_split(
                unique_keys,
                test_size=0.25,
                random_state=SEED,
                stratify=key_labels,
                shuffle=True,
            )

            tr_mask = np.isin(train_keys, keys_tr_split)
            val_mask = np.isin(train_keys, keys_val_split)

            Xtr_int, Xval_int = X_train_int[tr_mask], X_train_int[val_mask]
            Xtr_ext, Xval_ext = X_train_ext[tr_mask], X_train_ext[val_mask]
            ytr, yval = y_train[tr_mask], y_train[val_mask]

            if Xtr_int.shape != Xtr_ext.shape:
                raise RuntimeError(
                    f"Misaligned internal/external training: {Xtr_int.shape} vs"
                    f" {Xtr_ext.shape}"
                )
            if Xval_int.shape != Xval_ext.shape:
                raise RuntimeError(
                    f"Misaligned internal/external validation: {Xval_int.shape} vs"
                    f" {Xval_ext.shape}"
                )

            drills_val_arr = drills_train[val_mask]
            holes_val_arr = holes_train[val_mask]

            if len(np.unique(ytr)) < 2 or len(np.unique(yval)) < 2:
                print(
                    f"[NOTICE] Few classes in the drill split {drill_test}."
                )
                continue

            classes = np.unique(ytr)
            cw = dict(
                zip(
                    classes,
                    compute_class_weight(
                        class_weight="balanced", classes=classes, y=ytr
                    ),
                )
            )

            input_shape = Xtr_int.shape[1:]

            reset_random_seeds()
            model_fusion = build_late_fusion_architecture(input_shape=input_shape)
            model_fusion.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=["accuracy"],
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )

            model_fusion.fit(
                [Xtr_int, Xtr_ext],
                ytr,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=([Xval_int, Xval_ext], yval),
                callbacks=[early_stop],
                class_weight=cw,
                shuffle=True,
                verbose=0,
            )

            val_probs = model_fusion.predict(
                [Xval_int, Xval_ext], verbose=0
            ).flatten()

            val_keys = np.asarray(
                [f"{d}__{h}" for d, h in zip(drills_val_arr, holes_val_arr)],
                dtype=object,
            )
            u_val_keys = np.unique(val_keys)
            val_scores_h = np.array(
                [np.mean(val_probs[val_keys == k]) for k in u_val_keys],
                dtype=np.float32,
            )
            val_labels_h = np.array(
                [int(np.max(yval[val_keys == k])) for k in u_val_keys], dtype=np.int32
            )

            candidates = np.unique(
                np.percentile(val_scores_h, np.linspace(1, 99, 200))
            )
            best_thresh = float(np.median(val_scores_h))
            best_f1 = -1.0

            for thresh in candidates:
                val_preds = (val_scores_h >= thresh).astype(int)
                score = f1_score(val_labels_h, val_preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thresh = float(thresh)

            test_probs = model_fusion.predict(
                [X_test_int, X_test_ext], verbose=0
            ).flatten()
            u_test_holes = np.sort(np.unique(holes_test))
            test_scores_h = np.array(
                [
                    np.mean(test_probs[holes_test == h])
                    for h in u_test_holes
                    if np.any(holes_test == h)
                ],
                dtype=np.float32,
            )
            test_labels_h = np.array(
                [
                    int(np.max(y_test[holes_test == h]))
                    for h in u_test_holes
                    if np.any(holes_test == h)
                ],
                dtype=np.int32,
            )

            test_preds_h = (test_scores_h >= best_thresh).astype(int)
            metrics = compute_metrics_custom(
                test_labels_h, test_preds_h, test_scores_h
            )

            all_results.append({
                "drill": drill_test,
                "domain": DOMAIN,
                "n_mics": MIC_CFG,
                "config": "Feature_Fusion",
                "threshold": best_thresh,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "auc": metrics["auc"],
                "mcc": metrics["mcc"],
            })

            probs_all = model_fusion.predict(
                [X_all_int, X_all_ext], verbose=0
            ).flatten()
            u_all_holes = np.sort(np.unique(holes_all))
            continuous_rows = []
            for h in u_all_holes:
                idx = holes_all == h
                continuous_rows.append({
                    "drill": drill_test,
                    "domain": DOMAIN,
                    "n_mics": MIC_CFG,
                    "config": "Feature_Fusion",
                    "hole": int(h),
                    "predicted_anomaly_score": float(np.mean(probs_all[idx])),
                    "is_jam": int(np.any(y_all_raw[idx] == 1)),
                    "threshold": float(best_thresh),
                })

            pd.DataFrame(continuous_rows).to_csv(
                os.path.join(
                    HOLE_SCORES_DIR,
                    f"continuous_fusion_{DOMAIN}_{MIC_CFG}_{drill_test}.csv",
                ),
                index=False,
            )

            del model_fusion
            gc.collect()
            clear_session()

df_all = pd.DataFrame(all_results)
out_file = os.path.join(RESULTS_DIR, "lodo_feature_fusion_20drills.csv")
df_all.to_csv(out_file, index=False)

summary_file = os.path.join(
    RESULTS_DIR, "summary_feature_fusion_table2.csv"
)
summary = (
    df_all.groupby(["domain", "n_mics", "config"])[["f1", "auc", "mcc"]]
    .mean()
    .round(3)
)
summary.to_csv(summary_file)
