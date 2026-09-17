# Acoustic Sensor Fusion for Tool Condition Monitoring in CNC Drilling

This repository contains the complete source code, dataset pipelines, and experimental evaluation for our research on multimodal acoustic condition monitoring in Industry 4.0. The study investigates spatial transducer redundancy, machine enclosure acoustic transmission, and dual-domain spectral sensing to detect incipient tool wear and anticipate catastrophic jamming events across complete tool life cycles in Computer Numerical Control (CNC) drilling.

---

### 📌 Project Overview

Progressive tool degradation in high-speed CNC drilling induces sudden tool jamming, resulting in catastrophic tool breakage, workpiece scrap, and unplanned downtime. This project introduces a deep learning framework based on separable 2D Convolutional Neural Networks (2D-CNN) that extracts representations from full-bandwidth Log-Mel spectrograms to detect pre-failure acoustic anomalies.

The experimental benchmark evaluates four core operational dimensions across 20 solid carbide drills:
* **Sensing Domains:** Standard audible acoustic sensing (96 kHz sampling rate, up to 48 kHz) vs. ultrasonic acoustic emission sensing (192 kHz sampling rate, up to 96 kHz).
* **Enclosure Barriers:** Workpiece-coupled internal sensing (proximal to the cutting zone) vs. external sensing (airborne transmission through machine shielding).
* **Spatial Transducer Redundancy:** Performance scaling from single-transducer configurations (1-MIC) to multi-transducer spatial arrays (3-MIC).
* **Learning Paradigms:** Single-transducer cross-sensor transfer (Internal-to-Internal, Internal-to-External, External-to-External, External-to-Internal) vs. dual-branch feature-level late fusion combining synchronized internal and external spectrogram representations.

---

### 🔒 Methodology & Validation Protocol

To ensure realistic industrial generalization and prevent cross-tool information leakage, all experiments follow a strict **Leave-One-Drill-Out (LODO)** protocol:
* **Strict Tool Isolation:** In each fold, 19 physical drills are used to train and validate the models, while the remaining unseen drill is reserved strictly for blind testing.
* **Failure-Aligned Evaluation Windows:** The healthy nominal baseline is established using the first 5 holes of each tool, whereas the pre-failure degraded state is defined by the terminal 20% of tool life. Actual jamming cycles are excluded from metric calculation to benchmark proactive warning capabilities.
* **Decoupled Threshold Calibration:** Classification thresholds are optimized exclusively on the source-domain validation fold via F1-score maximization and frozen prior to inference on the target drill.
* **Statistical Rigor:** All performance deltas are validated via global Friedman tests and post-hoc pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction and effect size ($r = \vert{}Z\vert{} / \sqrt{N}$).

---

### 📁 Repository Structure

```text
acoustic_sensor_fusion_tcm
│
├── data/
│   ├── raw/                        # Raw multi-channel audio recordings (.wav)
│   ├── standardized/               # Standardized single-channel audio files
│   ├── segmented/                  # Individual hole audio cycles segmented by energy valleys
│   ├── images/                     # 128x128 Log-Mel spectrogram images (.png)
│   ├── metadata/                   # Annotation logs, process metadata, and datalogger records
│   ├── cache/                      # Preprocessed multi-tool dataset cache (processed_dataset.npz)
│   └── plots/                      # Supplementary inspection plots
│
├── results/
│   ├── cross_sensor_hole_scores/   # Sequential test prediction trajectories (Cross-Sensor)
│   ├── feature_fusion_hole_scores/ # Sequential test prediction trajectories (Feature Fusion)
│   ├── lodo_cross_sensor_20drills.csv     # Raw LODO test metrics for Cross-Sensor setups
│   ├── lodo_feature_fusion_20drills.csv   # Raw LODO test metrics for Feature-Level Fusion
│   ├── summary_cross_sensor_table1.csv    # Consolidated averages for Cross-Sensor (Table I)
│   ├── summary_feature_fusion_table2.csv  # Consolidated averages for Feature Fusion (Table II)
│   ├── friedman_table3.csv                # Global non-parametric Friedman test results (Table III)
│   ├── wilcoxon_holm_1mic_vs_3mic.csv     # Pairwise analysis for transducer density (1-MIC vs 3-MIC)
│   └── wilcoxon_holm_cross_vs_fusion.csv  # Pairwise analysis for Cross-Sensor vs Feature Fusion
│
├── src/
│   ├── preprocessing/              # Data ingestion and signal preprocessing pipeline
│   │   ├── convert.py              # Audio standardization and multi-channel splitting
│   │   ├── segment.py              # Energy valley segmentation and individual hole isolation
│   │   ├── create_images.py        # Log-Mel spectrogram generation (128x128)
│   │   └── prepare_dataset.py      # Structural dataset auditing and NPZ compression
│   │
│   ├── models/                     # Deep learning neural network architectures
│   │   ├── cnn_branch.py           # Separable 2D-CNN feature extraction branch
│   │   └── hybrid_cnn_2d.py        # Single-sensor and dual-branch late fusion models
│   │
│   ├── analysis/                   # LODO evaluation loops and statistical inference
│   │   ├── comparison_cross_sensor.py         # Cross-sensor evaluation (II, IE, EE, EI)
│   │   ├── comparison_feature-level_fusion.py # Dual-branch feature-level late fusion pipeline
│   │   └── statistical_tests.py               # Friedman and Wilcoxon-Holm significance tests
│   │
│   └── visualization/              # Manuscript visualization scripts
│       ├── sensor_comparison_2x2.py # 2x2 Log-Mel spectrogram comparison (Fig. 4)
│       ├── plots.py                 # Boxplots and aggregated performance bars (Figs. 3, 4, 5)
│       └── plot_6.py                # Multi-drill sequential degradation trajectories (Fig. 6)
│
├── .gitignore                      # Git exclusion rules for large datasets, caches, and weights
├── .gitattributes                  # Git repository attributes
├── requirements.txt                # Python package dependencies
└── README.md                       # Repository documentation