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

from src.models.hybrid_cnn_2d import build_robust_cnn_architecture

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
CACHE_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, "..", "..", "data", "cache"))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, "..", "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_FILE = os.path.join(CACHE_DIR, "processed_dataset.npz")
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(f"NPZ file not found: {CACHE_FILE}")

loaded = np.load(CACHE_FILE, allow_pickle=True)
drills_data = loaded["drills_data"].item()
loaded.close()

ALL_DRILLS = sorted(list(drills_data.keys()))
DOMAINS = ["regular", "ultrasonic"]
MIC_CONFIGS = ["1mic", "3mic"]

ONE_MIC_EXTERNAL = [1]
THREE_MIC_EXTERNAL = [1, 2, 3]

ONE_MIC_INTERNAL = [4]
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
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0

    try:
        auc = 0.5 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, scores)
    except Exception:
        auc = 0.5

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "mcc": mcc,
    }

def learn_threshold_on_val(val_probs, val_drills, val_holes, val_labels):
    val_keys = np.asarray([f"{d}__{h}" for d, h in zip(val_drills, val_holes)], dtype=object)
    unique_val_keys = np.unique(val_keys)

    val_scores_hole = [np.mean(val_probs[val_keys == k]) for k in unique_val_keys]
    val_labels_hole = [int(np.max(val_labels[val_keys == k])) for k in unique_val_keys]

    val_scores_hole = np.asarray(val_scores_hole, dtype=np.float32)
    val_labels_hole = np.asarray(val_labels_hole, dtype=np.int32)

    candidates = np.unique(np.percentile(val_scores_hole, np.linspace(1, 99, 200)))
    best_thresh = float(np.median(val_scores_hole))
    best_f1 = -1.0

    for thresh in candidates:
        val_preds = (val_scores_hole >= thresh).astype(int)
        score = f1_score(val_labels_hole, val_preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    return best_thresh


def evaluate_on_test(test_probs, test_drills, test_holes, test_labels, thresh):
    test_keys = np.asarray([f"{d}__{h}" for d, h in zip(test_drills, test_holes)], dtype=object)
    unique_test_keys = np.unique(test_keys)

    test_scores_hole = [np.mean(test_probs[test_keys == k]) for k in unique_test_keys]
    test_labels_hole = [int(np.max(test_labels[test_keys == k])) for k in unique_test_keys]

    test_scores_hole = np.asarray(test_scores_hole, dtype=np.float32)
    test_labels_hole = np.asarray(test_labels_hole, dtype=np.int32)

    test_preds = (test_scores_hole >= thresh).astype(int)
    metrics = compute_metrics_custom(test_labels_hole, test_preds, test_scores_hole)

    return metrics, unique_test_keys, test_scores_hole, test_labels_hole, test_preds


def get_tr_data(data, tr, position, domain):
    X_key = f"X_TR{tr}_{position}_{domain}"
    y_key = f"y_TR{tr}_{position}_{domain}"
    holes_key = f"holes_TR{tr}_{position}_{domain}"
    return (
        np.asarray(data[X_key], dtype=np.float32),
        np.asarray(data[y_key], dtype=np.float32),
        np.asarray(data[holes_key], dtype=np.int32),
    )


def align_trs_by_hole(data, trs, position, domain):
    tr_data = {}
    for tr in trs:
        X, y, holes = get_tr_data(data, tr, position, domain)
        tr_data[tr] = {"X": X, "y": y, "holes": holes}

    common_holes = None
    for tr in trs:
        current_holes = set(int(h) for h in tr_data[tr]["holes"])
        common_holes = current_holes if common_holes is None else common_holes & current_holes

    common_holes = sorted(common_holes) if common_holes else []
    aligned = {}
    for tr in trs:
        info = tr_data[tr]
        hole_to_index = {int(h): idx for idx, h in enumerate(info["holes"])}
        indices = [hole_to_index[h] for h in common_holes]
        aligned[tr] = {
            "X": info["X"][indices],
            "y": info["y"][indices],
            "holes": np.asarray(common_holes, dtype=np.int32),
        }
    return aligned


def build_mic_configuration(data, domain, mic_cfg, position, filter_criteria=True):
    tr_list = (
        ONE_MIC_EXTERNAL
        if (position == "external" and mic_cfg == "1mic")
        else THREE_MIC_EXTERNAL
        if (position == "external" and mic_cfg == "3mic")
        else ONE_MIC_INTERNAL
        if (position == "internal" and mic_cfg == "1mic")
        else THREE_MIC_INTERNAL
    )

    aligned = align_trs_by_hole(data, tr_list, position, domain)

    if mic_cfg == "1mic":
        tr = tr_list[0]
        X_raw = aligned[tr]["X"]
        y_raw = aligned[tr]["y"]
        holes_raw = aligned[tr]["holes"]
    else:
        X_fused = np.mean(
            np.stack([aligned[tr]["X"] for tr in tr_list], axis=0),
            axis=0,
        ).astype(np.float32)
        ref_tr = tr_list[0]
        X_raw = X_fused
        y_raw = aligned[ref_tr]["y"]
        holes_raw = aligned[ref_tr]["holes"]

    if filter_criteria:
        X_f, y_f, holes_f = apply_paper_criteria(X_raw, y_raw, holes_raw)
        return {"X": X_f, "y": y_f, "holes": holes_f}
    else:
        return {"X": X_raw, "y": y_raw, "holes": holes_raw}


def build_position_dataset(drills_subset, domain, mic_cfg, position, filter_criteria=True):
    X_list, y_list, holes_list, drills_list = [], [], [], []
    for drill_name in sorted(drills_subset.keys()):
        info = build_mic_configuration(
            drills_subset[drill_name], domain, mic_cfg, position, filter_criteria=filter_criteria
        )
        if len(info["X"]) > 0:
            X_list.append(info["X"])
            y_list.append(info["y"])
            holes_list.append(info["holes"])
            drills_list.append(np.full(len(info["holes"]), drill_name, dtype=object))

    if len(X_list) == 0:
        raise RuntimeError(f"No data available for {position} | {domain} | {mic_cfg}")

    return (
        np.concatenate(X_list, axis=0).astype(np.float32),
        np.concatenate(y_list, axis=0).astype(np.float32),
        np.concatenate(holes_list, axis=0).astype(np.int32),
        np.concatenate(drills_list, axis=0),
    )


def align_cross_sensor_sets(
        X_int, y_int, holes_int, drills_int, X_ext, y_ext, holes_ext, drills_ext
):
    keys_int = np.asarray([f"{d}__{h}" for d, h in zip(drills_int, holes_int)], dtype=object)
    keys_ext = np.asarray([f"{d}__{h}" for d, h in zip(drills_ext, holes_ext)], dtype=object)

    common_keys = np.intersect1d(np.unique(keys_int), np.unique(keys_ext))
    mask_int = np.isin(keys_int, common_keys)
    mask_ext = np.isin(keys_ext, common_keys)

    X_int, y_int, holes_int, drills_int, keys_int = (
        X_int[mask_int],
        y_int[mask_int],
        holes_int[mask_int],
        drills_int[mask_int],
        keys_int[mask_int],
    )
    X_ext, y_ext, holes_ext, drills_ext, keys_ext = (
        X_ext[mask_ext],
        y_ext[mask_ext],
        holes_ext[mask_ext],
        drills_ext[mask_ext],
        keys_ext[mask_ext],
    )

    order_int = np.argsort(keys_int, kind="stable")
    order_ext = np.argsort(keys_ext, kind="stable")

    return (
        X_int[order_int],
        y_int[order_int],
        holes_int[order_int],
        drills_int[order_int],
        X_ext[order_ext],
        y_ext[order_ext],
        holes_ext[order_ext],
        drills_ext[order_ext],
    )


all_results = []
hole_scores_dir = os.path.join(RESULTS_DIR, "cross_sensor_hole_scores")
os.makedirs(hole_scores_dir, exist_ok=True)

for DOMAIN in DOMAINS:
    for MIC_CFG in MIC_CONFIGS:
        print(f"\n DOMAIN: {DOMAIN.upper()} | MICS: {MIC_CFG.upper()}")

        for drill_test in ALL_DRILLS:
            print(f" Drill Cuttings Processing Test: {drill_test}")
            clear_session()
            gc.collect()
            reset_random_seeds()

            train_drills = {d: drills_data[d] for d in ALL_DRILLS if d != drill_test}
            test_drills = {drill_test: drills_data[drill_test]}

            (
                X_train_int,
                y_train_int,
                holes_train_int,
                drills_train_int,
            ) = build_position_dataset(train_drills, DOMAIN, MIC_CFG, "internal", filter_criteria=True)
            (
                X_train_ext,
                y_train_ext,
                holes_train_ext,
                drills_train_ext,
            ) = build_position_dataset(train_drills, DOMAIN, MIC_CFG, "external", filter_criteria=True)

            (
                X_train_int,
                y_train_int,
                holes_train_int,
                drills_train_int,
                X_train_ext,
                y_train_ext,
                holes_train_ext,
                drills_train_ext,
            ) = align_cross_sensor_sets(
                X_train_int,
                y_train_int,
                holes_train_int,
                drills_train_int,
                X_train_ext,
                y_train_ext,
                holes_train_ext,
                drills_train_ext,
            )

            (
                X_test_int,
                y_test_int,
                holes_test_int,
                drills_test_int,
            ) = build_position_dataset(test_drills, DOMAIN, MIC_CFG, "internal", filter_criteria=True)
            (
                X_test_ext,
                y_test_ext,
                holes_test_ext,
                drills_test_ext,
            ) = build_position_dataset(test_drills, DOMAIN, MIC_CFG, "external", filter_criteria=True)

            (
                X_test_int,
                y_test_int,
                holes_test_int,
                drills_test_int,
                X_test_ext,
                y_test_ext,
                holes_test_ext,
                drills_test_ext,
            ) = align_cross_sensor_sets(
                X_test_int,
                y_test_int,
                holes_test_int,
                drills_test_int,
                X_test_ext,
                y_test_ext,
                holes_test_ext,
                drills_test_ext,
            )

           
            (
                X_all_int,
                y_jam_all_int,
                holes_all_int,
                drills_all_int,
            ) = build_position_dataset(test_drills, DOMAIN, MIC_CFG, "internal", filter_criteria=False)
            (
                X_all_ext,
                y_jam_all_ext,
                holes_all_ext,
                drills_all_ext,
            ) = build_position_dataset(test_drills, DOMAIN, MIC_CFG, "external", filter_criteria=False)

            (
                X_all_int,
                y_jam_all_int,
                holes_all_int,
                drills_all_int,
                X_all_ext,
                y_jam_all_ext,
                holes_all_ext,
                drills_all_ext,
            ) = align_cross_sensor_sets(
                X_all_int,
                y_jam_all_int,
                holes_all_int,
                drills_all_int,
                X_all_ext,
                y_jam_all_ext,
                holes_all_ext,
                drills_all_ext,
            )

            train_keys = np.asarray(
                [f"{d}__{h}" for d, h in zip(drills_train_int, holes_train_int)], dtype=object
            )
            unique_train_keys = np.unique(train_keys)
            key_labels = np.asarray(
                [int(np.max(y_train_int[train_keys == k])) for k in unique_train_keys],
                dtype=np.int32,
            )

            keys_tr_split, keys_val = train_test_split(
                unique_train_keys,
                test_size=0.25,
                random_state=SEED,
                stratify=key_labels,
                shuffle=True,
            )

            train_mask = np.isin(train_keys, keys_tr_split)
            val_mask = np.isin(train_keys, keys_val)

            Xtr_int, Xval_int = X_train_int[train_mask], X_train_int[val_mask]
            Xtr_ext, Xval_ext = X_train_ext[train_mask], X_train_ext[val_mask]
            ytr, yval = y_train_int[train_mask], y_train_int[val_mask]

            drills_val_arr = drills_train_int[val_mask]
            holes_val_arr = holes_train_int[val_mask]
            drills_test_arr = drills_test_int
            holes_test_arr = holes_test_int

            if len(np.unique(ytr)) < 2 or len(np.unique(yval)) < 2:
                print(f"Division without the two classes during the drill{drill_test}.")
                continue

            classes = np.unique(ytr)
            cw = dict(
                zip(
                    classes,
                    compute_class_weight(class_weight="balanced", classes=classes, y=ytr),
                )
            )
            if Xtr_int.shape[1:] != Xtr_ext.shape[1:]:
                raise RuntimeError(
                    f"Incompatible shapes: Internal={Xtr_int.shape[1:]}, "
                    f"External={Xtr_ext.shape[1:]}"
                )
            input_shape = Xtr_int.shape[1:]

            reset_random_seeds()
            model_internal = build_robust_cnn_architecture(input_shape=input_shape)
            model_internal.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=["accuracy"],
            )
            early_stop_internal = EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
            model_internal.fit(
                Xtr_int,
                ytr,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(Xval_int, yval),
                callbacks=[early_stop_internal],
                class_weight=cw,
                verbose=0,
            )

            reset_random_seeds()
            model_external = build_robust_cnn_architecture(input_shape=input_shape)
            model_external.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=["accuracy"],
            )
            early_stop_external = EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
            model_external.fit(
                Xtr_ext,
                ytr,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(Xval_ext, yval),
                callbacks=[early_stop_external],
                class_weight=cw,
                verbose=0,
            )

            probs_val_int = model_internal.predict(Xval_int, verbose=0).flatten()
            probs_val_ext = model_external.predict(Xval_ext, verbose=0).flatten()

            thresh_internal = learn_threshold_on_val(
                probs_val_int, drills_val_arr, holes_val_arr, yval
            )
            thresh_external = learn_threshold_on_val(
                probs_val_ext, drills_val_arr, holes_val_arr, yval
            )

            probs_test_int = model_internal.predict(X_test_int, verbose=0).flatten()
            probs_test_ext_trans = model_internal.predict(X_test_ext, verbose=0).flatten()
            probs_test_ext = model_external.predict(X_test_ext, verbose=0).flatten()
            probs_test_int_trans = model_external.predict(X_test_int, verbose=0).flatten()

            probs_all_int = model_internal.predict(X_all_int, verbose=0).flatten()
            probs_all_ext_trans = model_internal.predict(X_all_ext, verbose=0).flatten()
            probs_all_ext = model_external.predict(X_all_ext, verbose=0).flatten()
            probs_all_int_trans = model_external.predict(X_all_int, verbose=0).flatten()

            configs = [
                ("II", probs_test_int, thresh_internal, probs_all_int, y_test_int),
                ("IE", probs_test_ext_trans, thresh_internal, probs_all_ext_trans, y_test_ext),
                ("EE", probs_test_ext, thresh_external, probs_all_ext, y_test_ext),
                ("EI", probs_test_int_trans, thresh_external, probs_all_int_trans, y_test_int),
            ]

            for cfg_name, t_prob_metric, applied_thresh, t_prob_all, t_labels in configs:
                metrics, _, _, _, _ = evaluate_on_test(
                    t_prob_metric,
                    drills_test_arr,
                    holes_test_arr,
                    t_labels,
                    applied_thresh,
                )

                all_results.append({
                    "drill": drill_test,
                    "domain": DOMAIN,
                    "n_mics": MIC_CFG,
                    "config": cfg_name,
                    "threshold": applied_thresh,
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "auc": metrics["auc"],
                    "mcc": metrics["mcc"],
                })

                unique_all_holes = np.sort(np.unique(holes_all_int))
                continuous_rows = []
                for h in unique_all_holes:
                    idx = holes_all_int == h
                    continuous_rows.append({
                        "drill": drill_test,
                        "domain": DOMAIN,
                        "n_mics": MIC_CFG,
                        "config": cfg_name,
                        "hole": int(h),
                        "predicted_anomaly_score": float(np.mean(t_prob_all[idx])),
                        "is_jam": int(np.any(y_jam_all_int[idx] == 1)),
                        "threshold": float(applied_thresh),
                    })

                pd.DataFrame(continuous_rows).to_csv(
                    os.path.join(
                        hole_scores_dir,
                        f"continuous_{DOMAIN}_{MIC_CFG}_{cfg_name}_{drill_test}.csv",
                    ),
                    index=False,
                )

            del model_internal, model_external
            gc.collect()
            clear_session()


df_all = pd.DataFrame(all_results)
out_file = os.path.join(RESULTS_DIR, "lodo_cross_sensor_20drills.csv")
df_all.to_csv(out_file, index=False)

summary_file = os.path.join(RESULTS_DIR, "summary_cross_sensor_table1.csv")
summary = (
    df_all.groupby(["domain", "n_mics", "config"])[["f1", "auc", "mcc"]]
    .mean()
    .round(3)
)
summary.to_csv(summary_file)
