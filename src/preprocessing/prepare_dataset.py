import os
import re
from collections import Counter, defaultdict
import cv2
import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, "..", "..", "data"))

IMAGE_DIR = os.path.join(DATA_DIR, "images")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_FILE = os.path.join(CACHE_DIR, "processed_dataset.npz")

IMG_SIZE = 128

DEBUG = True

# If True, any critical audit error prevents the creation of the NPZ file.
FAIL_ON_CRITICAL_ERROR = True


# JAM EVENTS
#
# "relative":
#   each integer represents the increment to the next JAM.
#
# "absolute":
#   integers directly denote the absolute hole indices with JAM.

JAM_EVENTS = {
    "drill_4mm_04": {"relative": [39, 10, 3]},
    "drill_4mm_05": {"relative": [93, 3, 11]},
    "drill_4mm_06": {"relative": [22, 3, 1]},
    "drill_4mm_07": {"relative": [4, 27, 1, 1]},
    "drill_4mm_08": {"relative": [7, 9, 3, 3]},
    "drill_4mm_09": {"relative": [44, 6, 2]},
    "drill_4mm_10": {"absolute": [2, 9, 14, 50, 58, 60]},
    "drill_4mm_11": {"relative": [16, 3, 1]},
    "drill_4mm_12": {"relative": [16, 1, 1]},
    "drill_4mm_13": {"relative": [15, 2, 2]},
    "drill_4mm_14": {"relative": [13, 8, 1]},
    "drill_4mm_15": {"relative": [4, 3, 2, 5, 17, 6, 2, 4, 2]},
    "drill_4mm_16": {"absolute": [7, 14, 17, 29, 33, 38, 39]},
    "drill_4mm_17": {"relative": [1, 8, 4, 1, 2, 4]},
    "drill_4mm_18": {"absolute": [3, 45, 52, 68, 75, 78, 80, 82]},
    "drill_4mm_19": {"relative": [10, 38, 2, 3, 1]},
    "drill_4mm_20": {"relative": [5, 7, 12, 6, 2, 3, 1]},
    "drill_4mm_21": {"relative": [13, 9, 2, 2, 3]},
    "drill_4mm_22": {"relative": [22, 1, 1, 1]},
    "drill_4mm_23": {"relative": [26, 2, 5, 1, 5, 2]},
}

# JAM EVENTS CONVERSION

def relative_to_absolute(events):
  holes = []
  current = 0
  for event in events:
    current += event
    holes.append(current)
  return set(holes)


JAM_LOOKUP = {}

for drill, info in JAM_EVENTS.items():
  if "absolute" in info:
    JAM_LOOKUP[drill] = set(info["absolute"])
  elif "relative" in info:
    JAM_LOOKUP[drill] = relative_to_absolute(info["relative"])
  else:
    raise ValueError(f"Invalid JAM_EVENTS definition for {drill}")

# REGEX PATTERNS

FILE_REGEX = re.compile(
    r"Tr(?P<tr>\d+)_" r"(?P<position>ext|int)_" r"hole(?P<hole>\d+)",
    re.IGNORECASE,
)

DRILL_REGEX = re.compile(r"drill_4mm_(\d+)", re.IGNORECASE)


def debug_print(*args, **kwargs):
  if DEBUG:
    print(*args, **kwargs)


def critical(errors, message):
  errors.append(message)
  print("CRITICAL ERROR")
  print(message)


def warning(warnings, message):
  warnings.append(message)
  print("[WARNING]")
  print(message)


def empty_image_array():
  return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)


def load_image(filename):
  img = cv2.imread(filename)
  if img is None:
    raise FileNotFoundError(f"Failed to open image file:\n{filename}")
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
  img = img.astype(np.float32) / 255.0
  return img


def extract_drill_from_path(path):
  for part in path.split(os.sep):
    match = DRILL_REGEX.search(part)
    if match:
      return f"drill_4mm_{match.group(1)}"
  return None


def extract_file_information(filename):
  match = FILE_REGEX.search(filename)
  if match is None:
    return None

  tr = int(match.group("tr"))
  position = match.group("position").lower()
  hole = int(match.group("hole"))

  # TR1, TR2, TR3 -> external
  # TR4, TR5, TR6 -> internal
  if tr in [1, 2, 3]:
    sensor_position = "external"
  elif tr in [4, 5, 6]:
    sensor_position = "internal"
  else:
    raise ValueError(f"Unexpected transducer index: TR{tr}")

  return {
      "tr": tr,
      "position": position,
      "hole": hole,
      "sensor_position": sensor_position,
  }


critical_errors = []
warnings = []
seen_paths = set()


dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


total_png = 0
valid_png = 0
invalid_png = 0

source_counter = Counter()
tr_counter = Counter()
position_counter = Counter()


if not os.path.exists(IMAGE_DIR):
  raise FileNotFoundError(f"Image directory not found:\n{IMAGE_DIR}")


for root, _, files in os.walk(IMAGE_DIR):
  pngs = sorted(f for f in files if f.lower().endswith(".png"))
  if not pngs:
    continue

  root_lower = root.lower()

  # Identify sensing domain
  if "reg_mics" in root_lower:
    source = "regular"
  elif "ultrasonic_mics" in root_lower:
    source = "ultrasonic"
  else:
    continue

  # Identify drill identifier
  drill = extract_drill_from_path(root)
  if drill is None:
    critical(critical_errors, f"Could not identify drill from path:\n{root}")
    continue

  # Verify JAM lookup availability
  if drill not in JAM_LOOKUP:
    warning(warnings, f"Drill {drill} does not have JAM_EVENTS metadata.")

  # Process individual files
  for filename in pngs:
    total_png += 1
    fullpath = os.path.join(root, filename)

    # Avoid duplicate file paths
    normalized_path = os.path.normcase(os.path.abspath(fullpath))
    if normalized_path in seen_paths:
      critical(critical_errors, f"Duplicate file found:\n{fullpath}")
      continue
    seen_paths.add(normalized_path)

    # Extract naming metadata
    try:
      info = extract_file_information(filename)
    except Exception as error:
      critical(
          critical_errors,
          f"Error parsing file name:\n{fullpath}\n{error}",
      )
      invalid_png += 1
      continue

    if info is None:
      invalid_png += 1
      critical(
          critical_errors,
          "Invalid file name formatting:\n"
          f"{fullpath}\n\n"
          "Expected format:\n"
          "04_common_Tr1_ext_hole00001.png\n"
          "or\n"
          "04_common_Tr4_int_hole00001.png",
      )
      continue

    valid_png += 1

    tr = info["tr"]
    position_from_filename = info["position"]
    hole = info["hole"]
    sensor_position = info["sensor_position"]

    # Transducer-position consistency checks
    expected_position = "external" if tr <= 3 else "internal"
    expected_filename_position = "ext" if expected_position == "external" else "int"

    if position_from_filename != expected_filename_position:
      critical(
          critical_errors,
          "Transducer/position mismatch:\n"
          f"File     : {fullpath}\n"
          f"TR       : {tr}\n"
          f"Position : {position_from_filename}\n"
          f"Expected : {expected_filename_position}",
      )
      continue

    if sensor_position != expected_position:
      critical(
          critical_errors,
          "Sensor/transducer mismatch:\n"
          f"File     : {fullpath}\n"
          f"TR       : {tr}\n"
          f"Sensor   : {sensor_position}\n"
          f"Expected : {expected_position}",
      )
      continue

    # Binary label mapping
    if drill in JAM_LOOKUP:
      label = int(hole in JAM_LOOKUP[drill])
    else:
      label = -1

    relative_path = os.path.relpath(fullpath, IMAGE_DIR)

    # Load image content
    try:
      img = load_image(fullpath)
    except Exception as error:
      critical(critical_errors, f"Error loading image:\n{fullpath}\n{error}")
      continue

    # Detect duplicate samples within same drill/domain/TR/hole
    if hole in dataset[drill][source][tr]:
      previous = dataset[drill][source][tr][hole]
      critical(
          critical_errors,
          "SAMPLE DUPLICATION DETECTED!\n\n"
          f"Drill  : {drill}\n"
          f"Source : {source}\n"
          f"TR     : {tr}\n"
          f"Hole   : {hole}\n\n"
          f"Previous file:\n{previous['path']}\n\n"
          f"New file:\n{fullpath}",
      )
      continue

    # Store valid sample
    dataset[drill][source][tr][hole] = {
        "drill": drill,
        "source": source,
        "tr": tr,
        "position": expected_position,
        "hole": hole,
        "label": label,
        "img": img,
        "path": fullpath,
        "relative_path": relative_path,
    }

    source_counter[source] += 1
    tr_counter[(source, tr)] += 1
    position_counter[(source, expected_position)] += 1

# PARSING SUMMARY

print("FILE PARSING SUMMARY")
print(f"PNG files found   : {total_png}")
print(f"Valid PNG files   : {valid_png}")
print(f"Invalid PNG files : {invalid_png}")
print(f"Regular domain    : {source_counter['regular']}")
print(f"Ultrasonic domain : {source_counter['ultrasonic']}")

# TRANSDUCER AUDIT (TR1..TR6)


for drill in sorted(dataset):
  print()
  print(drill)

  for source in ["regular", "ultrasonic"]:
    print()
    print(f"  {source.upper()}")

    for tr in range(1, 7):
      samples = dataset[drill][source][tr]
      holes = sorted(samples.keys())
      expected_position = "external" if tr <= 3 else "internal"

      print(
          f"    TR{tr} ({expected_position:8s}) | "
          f"samples={len(samples):4d} | "
          f"holes={holes[0] if holes else '-'}-{holes[-1] if holes else '-'}"
      )

      if len(samples) == 0:
        critical(
            critical_errors,
            f"{drill}/{source}/TR{tr}: no samples found.",
        )


# LABEL INTEGRITY AUDIT

print()
print("=" * 80)
print("PHASE 4 — LABEL INTEGRITY AUDIT")
print("=" * 80)

for drill in sorted(dataset):
  if drill not in JAM_LOOKUP:
    continue

  expected_jams = sorted(JAM_LOOKUP[drill])
  print()
  print(drill)
  print(f"Expected jams: {expected_jams}")

  observed = {}
  for source in dataset[drill]:
    for tr in dataset[drill][source]:
      for sample in dataset[drill][source][tr].values():
        hole = sample["hole"]
        label = sample["label"]
        if hole not in observed:
          observed[hole] = set()
        observed[hole].add(label)

  inconsistent_holes = [
      hole for hole, labels in observed.items() if len(labels) != 1
  ]

  if inconsistent_holes:
    critical(
        critical_errors,
        f"{drill}: holes with inconsistent labels: {inconsistent_holes}",
    )

  wrong_labels = []
  for hole, labels in observed.items():
    expected_label = int(hole in JAM_LOOKUP[drill])
    for label in labels:
      if label != expected_label:
        wrong_labels.append((hole, expected_label, label))

  if wrong_labels:
    critical(critical_errors, f"{drill}: incorrect labels:\n{wrong_labels}")
  else:
    print("  OK — labels match JAM_EVENTS lookup.")

# CROSS-DOMAIN SENSOR CORRESPONDENCE

print()
print("=" * 80)
print("PHASE 5 — SENSOR CORRESPONDENCE AUDIT")
print("=" * 80)

for drill in sorted(dataset):
  print()
  print(f"### {drill}")

  for tr in range(1, 7):
    regular_samples = dataset[drill]["regular"][tr]
    ultrasonic_samples = dataset[drill]["ultrasonic"][tr]

    regular_holes = set(regular_samples.keys())
    ultrasonic_holes = set(ultrasonic_samples.keys())

    only_regular = regular_holes - ultrasonic_holes
    only_ultrasonic = ultrasonic_holes - regular_holes
    common = regular_holes & ultrasonic_holes

    print(
        f"TR{tr} | regular={len(regular_holes):3d} | "
        f"ultrasonic={len(ultrasonic_holes):3d} | common={len(common):3d} | "
        f"only_reg={len(only_regular):3d} | only_ult={len(only_ultrasonic):3d}"
    )

    if only_regular:
      warning(
          warnings,
          f"{drill} TR{tr}: holes present only in regular: {sorted(only_regular)}",
      )

    if only_ultrasonic:
      warning(
          warnings,
          f"{drill} TR{tr}: holes present only in ultrasonic: {sorted(only_ultrasonic)}",
      )

    for hole in sorted(common):
      reg_label = regular_samples[hole]["label"]
      ult_label = ultrasonic_samples[hole]["label"]
      if reg_label != ult_label:
        critical(
            critical_errors,
            f"{drill}/TR{tr}/hole{hole}: labels differ across sensing domains. "
            f"regular={reg_label}, ultrasonic={ult_label}",
        )

# CONSTRUCTING NPZ SCHEMA

print()
print("=" * 80)
print("PHASE 6 — BUILDING NPZ SCHEMA")
print("=" * 80)

drills_data = {}

for drill in sorted(dataset):
  print()
  print("-" * 80)
  print(drill)
  print("-" * 80)

  data = {}

  # 6.1 Individual arrays per transducer
  for source in ["regular", "ultrasonic"]:
    for tr in range(1, 7):
      position = "external" if tr <= 3 else "internal"
      samples = dataset[drill][source][tr]
      ordered_samples = sorted(
          samples.values(),
          key=lambda sample: (sample["hole"], sample["path"]),
      )

      images = [sample["img"] for sample in ordered_samples]
      labels = [sample["label"] for sample in ordered_samples]
      holes = [sample["hole"] for sample in ordered_samples]

      prefix = f"TR{tr}_{position}_{source}"

      if images:
        X = np.asarray(images, dtype=np.float32)
      else:
        X = empty_image_array()

      y = np.asarray(labels, dtype=np.int32)
      h = np.asarray(holes, dtype=np.int32)

      data[f"X_{prefix}"] = X
      data[f"y_{prefix}"] = y
      data[f"holes_{prefix}"] = h

      print(
          f"{prefix:35s} X={X.shape} y={y.shape} holes={h.shape}"
      )

  # 6.2 Aggregated arrays by position and domain
  for source in ["regular", "ultrasonic"]:
    for position, trs in [("external", [1, 2, 3]), ("internal", [4, 5, 6])]:
      all_samples = []
      for tr in trs:
        all_samples.extend(dataset[drill][source][tr].values())

      # Deterministic ordering: hole first, transducer as tiebreaker
      all_samples = sorted(
          all_samples,
          key=lambda sample: (sample["hole"], sample["tr"], sample["path"]),
      )

      X = np.asarray([sample["img"] for sample in all_samples], dtype=np.float32)
      y = np.asarray([sample["label"] for sample in all_samples], dtype=np.int32)
      holes = np.asarray([sample["hole"] for sample in all_samples], dtype=np.int32)

      key_prefix = f"{position}_{source}"
      data[f"X_{key_prefix}"] = X
      data[f"y_{key_prefix}"] = y
      data[f"holes_{key_prefix}"] = holes

      print(
          f"{key_prefix:35s} X={X.shape} y={y.shape} holes={holes.shape}"
      )

  # 6.3 Aggregated arrays across positions
  for position in ["external", "internal"]:
    regular_X = data[f"X_{position}_regular"]
    ultrasonic_X = data[f"X_{position}_ultrasonic"]
    regular_y = data[f"y_{position}_regular"]
    ultrasonic_y = data[f"y_{position}_ultrasonic"]
    regular_holes = data[f"holes_{position}_regular"]
    ultrasonic_holes = data[f"holes_{position}_ultrasonic"]

    X_position = np.vstack([regular_X, ultrasonic_X])
    y_position = np.concatenate([regular_y, ultrasonic_y])
    holes_position = np.concatenate([regular_holes, ultrasonic_holes])

    data[f"X_{position}"] = X_position
    data[f"y_{position}"] = y_position
    data[f"holes_{position}"] = holes_position

  # 6.4 Compatibility aliases
  data["X_external"] = data["X_external"]
  data["X_internal"] = data["X_internal"]
  data["y_external"] = data["y_external"]
  data["y_internal"] = data["y_internal"]
  data["holes_external"] = data["holes_external"]
  data["holes_internal"] = data["holes_internal"]

  # 6.5 Legacy aliases
  data["y"] = data["y_internal"]
  data["holes"] = data["holes_internal"]

  drills_data[drill] = data


for drill in sorted(drills_data):
  data = drills_data[drill]
  print()
  print(f"### {drill}")

  # Check individual arrays
  for source in ["regular", "ultrasonic"]:
    for tr in range(1, 7):
      position = "external" if tr <= 3 else "internal"
      prefix = f"TR{tr}_{position}_{source}"
      X = data[f"X_{prefix}"]
      y = data[f"y_{prefix}"]
      holes = data[f"holes_{prefix}"]

      if not (len(X) == len(y) == len(holes)):
        critical(
            critical_errors,
            f"{drill}/{prefix}: length mismatch: X={len(X)}, y={len(y)}, holes={len(holes)}",
        )

      if X.ndim != 4:
        critical(
            critical_errors,
            f"{drill}/{prefix}: X has ndim={X.ndim}; expected ndim=4.",
        )
      elif X.shape[1:] != (IMG_SIZE, IMG_SIZE, 3):
        critical(
            critical_errors,
            f"{drill}/{prefix}: shape={X.shape}; unexpected dimensions.",
        )

      if len(X) and np.isnan(X).any():
        critical(critical_errors, f"{drill}/{prefix}: NaN values detected.")

      if len(X) and np.isinf(X).any():
        critical(critical_errors, f"{drill}/{prefix}: Inf values detected.")

      invalid_labels = set(y.tolist()) - {0, 1}
      if invalid_labels:
        critical(
            critical_errors,
            f"{drill}/{prefix}: invalid label values found: {invalid_labels}",
        )

      print(
          f"{prefix:35s} X={X.shape} y={y.shape} holes={holes.shape}"
      )

  # Check aggregated position arrays
  for position in ["external", "internal"]:
    X = data[f"X_{position}"]
    y = data[f"y_{position}"]
    holes = data[f"holes_{position}"]

    if not (len(X) == len(y) == len(holes)):
      critical(
          critical_errors,
          f"{drill}/{position}: aggregated mismatch: X={len(X)}, y={len(y)}, holes={len(holes)}",
      )

    print(
        f"{position:35s} X={X.shape} y={y.shape} holes={holes.shape}"
    )

# DUPLICATION AUDIT BY TRANSDUCER AND HOLE

print()
print("=" * 80)
print("PHASE 8 — TRANSDUCER HOLE DUPLICATION AUDIT")
print("=" * 80)

for drill in sorted(dataset):
  for source in ["regular", "ultrasonic"]:
    for tr in range(1, 7):
      seen_holes = Counter()
      for sample in dataset[drill][source][tr].values():
        seen_holes[sample["hole"]] += 1

      duplicates = {
          hole: count for hole, count in seen_holes.items() if count > 1
      }
      if duplicates:
        critical(
            critical_errors,
            f"{drill}/{source}/TR{tr}: duplicated hole indices: {duplicates}",
        )


for drill in sorted(drills_data):
  data = drills_data[drill]
  print()
  print(f"### {drill}")

  checked_keys = []
  for key in data:
    if not key.startswith("X_"):
      continue
    if key in checked_keys:
      continue
    checked_keys.append(key)

    X = data[key]
    if X.ndim != 4 or len(X) == 0:
      continue

    min_value = X.min()
    max_value = X.max()
    mean_value = X.mean()

    print(
        f"{key:40s} min={min_value:.6f} max={max_value:.6f} mean={mean_value:.6f}"
    )

    if min_value < 0:
      critical(critical_errors, f"{drill}/{key}: min value < 0: {min_value}")
    if max_value > 1:
      critical(critical_errors, f"{drill}/{key}: max value > 1: {max_value}")



print("FINAL AUDIT REPORT")
print(f"PNG files found   : {total_png}")
print(f"Valid PNG files   : {valid_png}")
print(f"Invalid PNG files : {invalid_png}")
print(f"Critical errors   : {len(critical_errors)}")
print(f"Warnings          : {len(warnings)}")

if critical_errors:
  print("CRITICAL ERRORS FOUND:")
  for error in critical_errors[:20]:
    print("-", error)

if critical_errors and FAIL_ON_CRITICAL_ERROR:
  raise RuntimeError("Dataset audit failed. Fix errors prior to generating NPZ.")


np.savez_compressed(CACHE_FILE, drills_data=drills_data)

print("Dataset successfully saved to:")
print(CACHE_FILE)
