# Practical, private, and instance-dependent optimization algorithm for  Geometric Median

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue.svg)](https://neurips.cc/virtual/2024/poster/94421)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07407-b31b1b.svg)](https://arxiv.org/abs/2406.07407)


Official implementation of **"Private Geometric Median"** presented at NeurIPS 2024.

**Authors:** Mahdi Haghifam, Thomas Steinke, Jonathan Ullman

## Abstract

This repository contains the implementation of differentially private (DP) algorithms for computing the geometric median of a dataset. Given $n$ points $x_1, \ldots, x_n \in \mathbb{R}^d$, our goal is to find a point $\theta$ that minimizes the sum of Euclidean distances to these points while preserving privacy.

Unlike standard approaches like DP-GD that require strong a priori knowledge about data location within a ball of radius $R$ (with excess risk depending linearly on $R$), our algorithms provide excess error guarantees that scale with the **effective diameter** of the datapoints—the unknown radius containing the majority of the data.

| <img src="low-eps.jpg" width="400"> | <img src="high-eps.jpg" width="400"> |


## Installation & Running
```bash
# Clone the repository
git clone https://github.com/mhaghifam/Private-Geometric-Median-NeurIPS2024.git
cd Private-Geometric-Median-NeurIPS2024

# Install dependencies
pip install -r requirements.txt

# Run experiments to reproduce figures
python run_experiments.py

## Contact

For questions and feedback:
- **Mahdi Haghifam** - [haghifam.mahdi@gmail.com](mailto:haghifam.mahdi@gmail.com)



---

**Note:** This is research code. While we have tested it thoroughly, it may contain bugs. Please report any issues you encounter.
