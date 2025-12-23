<div align="center">

# 🏥 PASS: Probabilistic Agentic Supernet Sampling for Interpretable and Adaptive Chest X-Ray Reasoning

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.10501)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2508.10501-blue.svg)](https://doi.org/10.48550/arXiv.2508.10501)

[**Paper**](https://arxiv.org/abs/2508.10501) | [**Code**](https://github.com/ys-feng/PASS) | [**Data**](#) | [**Demo**](#)

</div>

---

## 🌟 Overview

**PASS** (Probabilistic Agentic Supernet Sampling) explores multimodal, adaptive, trustworthy agentic workflows for Chest X-Ray (CXR) reasoning.

- 🧠 **Adaptive Multimodal Decision-Making**: Multimodal task-conditioned tool selection over a multi-agentic supernet of medical tools
- 🎯 **Interpretable Workflows**: Probability-annotated paths for post-hoc inspection and safety review
- ⚡ **Efficiency-Aware Training**: Cost-sensitive learning with dynamic early exit
- 🔄 **Evolving Memory**: Compression of salient findings into a lightweight memory buffer
- 🧪 **CAB-E Benchmark**: A benchmark for multi-step, safety-critical, free-form CXR reasoning

📄 **Paper**: [arXiv:2508.10501](https://arxiv.org/abs/2508.10501) | 📖 **PDF**: [Download](https://arxiv.org/pdf/2508.10501)

---

## 🏗️ Framework

<div align="center">
  <img src="readme_fig/Framework.png" alt="PASS Framework" width="90%">
  <p><em>Figure 1: The PASS framework architecture.</em></p>
</div>

The PASS framework consists of three main components:

1. **Supernet $\mathcal{G}$**: A multi-tool graph containing specialized medical reasoning tools (e.g., visual grounding, region analysis, medical knowledge retrieval)
2. **Policy Network $\pi_\theta$**: Learns task-conditioned distribution to adaptively select tools at each supernet layer, providing probability-annotated trajectories
3. **Answer Generator $p_\phi$**: Produces final answer based on the key investigation results, reasoning trajectory and personalized memory

### Key Features

- ✅ To the best of our knowledge, PASS is the first multimodal agentic framework for CXR reasoning with interpretable probability annotations
- ✅ Three-stage training: Expert warm-up → Contrastive path-ranking → Cost-aware RL
- ✅ Dynamic early exit mechanism for computational efficiency
- ✅ Trustworthy-critical design with transparent decision paths for medical auditing
- ✅ CAB-E benchmark: New comprehensive evaluation for multi-step CXR reasoning

---

## 📦 Data

- **CAB-Standard**: Constructed using the ChestAgentBench (CAB) methodology introduced in the MedRAX paper and adapted to the SLAKE dataset. The original CAB construction process is detailed in the MedRAX paper and its benchmark generation script:
  - MedRAX paper (CAB method): [link](https://arxiv.org/html/2502.02673v1)
  - CAB generation script: [link](https://github.com/bowang-lab/MedRAX/blob/main/benchmark/create_benchmark.py)
- **SLAKE**: [link](https://www.med-vqa.com/slake/)

---

## 📝 Citation

If you find our work helpful, please consider citing our paper:
🔗 [Paper Link](https://arxiv.org/abs/2508.10501) | 📖 [PDF](https://arxiv.org/pdf/2508.10501)

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
