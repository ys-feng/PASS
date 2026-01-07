<div align="center">

# 🏥 PASS: Probabilistic Agentic Supernet Sampling for Interpretable and Adaptive Chest X-Ray Reasoning

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.10501)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)

[**Paper**](https://arxiv.org/abs/2508.10501) | [**Code**](https://github.com/ys-feng/PASS)

</div>

---

## Overview

**PASS** (Probabilistic Agentic Supernet Sampling) is a multimodal agentic framework for Chest X-Ray (CXR) reasoning. The framework features:

- **Adaptive Multimodal Tool Selection**: Task-conditioned policy network for dynamic tool orchestration
- **Interpretable Agentic Workflows**: Probability-annotated reasoning paths for clinical transparency and trustworthiness
- **Efficiency-Aware Learning**: Cost-sensitive training with dynamic early exit
- **Three-Stage Training**: Expert knowledged guided warm-up → Contrastive path-ranking → Cost-aware RL

---

## Framework

<div align="center">
  <img src="assets/Framework.png" alt="PASS Framework" width="90%">
  <p><em>The PASS framework.</em></p>
</div>

PASS employs a **probabilistic controller** to sample actions from an **agentic supernet** (DAG of medical agents), generating workflows with **interpretable probabilities** for clinical safety. Tool outputs are aggregated into **personalized memory** to inform subsequent steps. Trained via a three-stage strategy, PASS enables **interpretable, adaptive, and efficient** multimodal CXR reasoning.

---

## Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate pass
```

---

## Data

We provide three benchmarks for evaluation:

### CAB-E (Ours)

**CAB-E** (ChestAgentBench-E) is our proposed benchmark for multi-step, safety-critical, free-form CXR reasoning.

### CAB-Standard

Constructed using the **ChestAgentBench (CAB)** methodology introduced in MedRAX and adapted to the SLAKE dataset.

- MedRAX paper (CAB method): [arXiv:2502.02673](https://arxiv.org/html/2502.02673v1)
- CAB generation script: [MedRAX/benchmark](https://github.com/bowang-lab/MedRAX/blob/main/benchmark/create_benchmark.py)

### SLAKE

The benchmarks are built upon the SLAKE dataset. Please download images from the [official site](https://www.med-vqa.com/slake/) and place them in `data/Slake1.0/`.

---

## Usage

```bash
# Training
python main.py train

# Evaluation  
python main.py evaluate --checkpoint <path_to_checkpoint>

# See all options
python main.py --help
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{feng2025pass,
      title={PASS: Probabilistic Agentic Supernet Sampling for Interpretable and Adaptive Chest X-Ray Reasoning}, 
      author={Yushi Feng and Junye Du and Yingying Hong and Qifan Wang and Lequan Yu},
      year={2025},
      eprint={2508.10501},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.10501}, 
}
```

---

## Acknowledgement

We thank the [SLAKE](https://www.med-vqa.com/slake/) dataset creators for their contributions.

We also thank to the following repositories for their invaluable code and insights:

Our benchmark design is partially adapted from [MedRAX](https://github.com/bowang-lab/MedRAX). Our agentic framework and tool implementations are partially adapted from [MaAS](https://github.com/bingreeky/MaAS).

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
