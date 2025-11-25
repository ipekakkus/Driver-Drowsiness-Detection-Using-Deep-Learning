Driver Drowsiness Detection Using Deep Learning

This project focuses on developing a robust driver drowsiness detection system using state-of-the-art deep learning models. The goal is to identify drowsy vs non-drowsy driver states from facial images, evaluate model performance across datasets, and analyze generalization capability beyond a single environment.

The project includes:

🚗 Project Overview

Drowsy driving is a major cause of traffic accidents worldwide. The purpose of this project is to design and evaluate a vision-based detection system that can reliably determine whether a driver is drowsy in real time.

The project:
- Implements and trains multiple deep learning architectures
- Performs extensive performance evaluation
- Tests cross-dataset generalization
- Benchmarks inference speed and model efficiency
- Provides comparison visualizations and analysis
- All experiments, models, evaluation scripts, and dataset handling pipelines are included.

🧠 Models Implemented & Compared

The following CNN architectures were trained and evaluated:
- ResNet50V2
- MobileNetV2
- EfficientNetB0
- NASNetMobile

Each model was:
- fine-tuned on the NTHU-DDD dataset
- evaluated on both NTHU-DDD test split and full DDD dataset
- compared using multiple metrics and generalization plots

📦 Datasets Used
1. NTHU-DDD (NTHU Driver Drowsiness Detection Dataset)
   Used for training, validation, and in-distribution testing.

2. DDD (Driver Drowsiness Dataset)
  Used for cross-dataset generalization experiments.
  Contains 41,000+ labeled images categorized into:
  - Drowsy
  - Non Drowsy
    
This dataset helps test whether a model trained on NTHU-DDD generalizes to a different environment, lighting condition, and subject pool.

📊 Evaluation Metrics

Each model is evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC (ROC-AUC)

Additionally, the project includes:

- Confusion Matrices
- ROC Curves
- Side-by-side metric comparisons
- Generalization drop charts (Δ between datasets)
- Multi-metric summary visualizations
- Radar charts showing metric “shape” differences
- Inference speed (images/sec)
- Total evaluation time per model

🚀 Key Features of This Repository
✔ Complete training pipeline

- Training scripts for all four models using TensorFlow / Keras.

✔ Cross-dataset generalization analysis

Full evaluation on two datasets with detailed visualizations.

✔ Reproducible experiments

All scripts follow the same pipeline structure and can be re-run easily.

Driver-Drowsiness-Detection/
│
├── models/
│   ├── resnet50v2_final.keras
│   ├── mobilenetv2_final.keras
│   ├── efficientnetb0_final.keras
│   └── nasnetmobile_final.keras
│
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_training_pipeline.ipynb
│   ├── 3_cross_dataset_evaluation.ipynb
│   └── model_comparison.ipynb   <-- this is your main comparison notebook
│
├── scripts/
│   ├── dataset_loader.py
│   ├── training_script.py
│   ├── evaluation_utils.py
│   └── visualization_utils.py
│
├── results/
│   ├── metrics/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── generalization_plots/
│   └── timing_summary.csv
│
└── README.md


