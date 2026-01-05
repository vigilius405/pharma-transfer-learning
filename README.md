# pharma-transfer-learning
Final project for Distributed Optimization.

Running pharma_ftl.ipynb will guide the training of one party, all-reduce, FedBCD, and FedCVT models (toy classifiers are small), and will enable recreation of figures/tables produced as project results. For setup run `pip install -r requirements.txt`.

# Overview

This repository implements and evaluates vertical federated learning (VFL) and federated transfer learning (FTL) methods on small-molecule classification tasks relevant to drug discovery. The project addresses the real-world challenge of data silos across pharmaceutical organizations, where datasets differ in both features and samples and cannot be shared directly.

Using MoleculeNet benchmarks, the study demonstrates that FedBCD and FedCVT can match or outperform centralized models, even in low-data, heterogeneous settings.

# Key Contributions
- Practical evaluation of VFL and FTL algorithms on biochemical data
- Benchmarking against centralized and horizontal federated baselines
- Demonstration that distributed learning can outperform an “omniscient” single-party model
- Realistic simulation of pharmaceutical data fragmentation

# Learning Paradigms Implemented
- Centralized (one-party) learning
- Horizontal federated learning (All-Reduce)
- Vertical federated learning (FedBCD)
- Federated transfer learning (FedCVT)

# Datasets
MoleculeNet classification tasks:
- ClinTox
- Tox21
- BBBP
- SIDER
- ToxCast
Information on datasets here: https://moleculenet.org/datasets-1

# Referenced Algorithms
- FedBCD: Block Coordinate Descent for vertical federated learning
- FedCVT: Cross-view training for semi-supervised federated transfer learning
Information on UME here: https://huggingface.co/karina-zadorozhny/ume-base

# Results
- FedBCD and FedCVT achieve significant accuracy improvements over centralized learning
- FedCVT effectively leverages unlabeled and non-overlapping samples
- Distributed models benefit from increased effective parameterization

# Use Cases
- Privacy-preserving collaboration in drug discovery
- Learning from fragmented, proprietary biochemical datasets
- Low-label and heterogeneous scientific ML settings
