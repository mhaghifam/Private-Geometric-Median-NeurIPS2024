# Private Geometric Median

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue.svg)](https://neurips.cc/virtual/2024/poster/94421)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07407-b31b1b.svg)](https://arxiv.org/abs/2406.07407)


Official implementation of **"Private Geometric Median"** presented at NeurIPS 2024.

**Authors:** [Mahdi Haghifam](https://mhaghifam.github.io/mahdihaghifam/), [Thomas Steinke](https://scholar.google.com/citations?user=sLW5VVIAAAAJ), [Jonathan Ullman](https://www.ccs.neu.edu/home/jullman/)

## Abstract

This repository contains the implementation of differentially private (DP) algorithms for computing the geometric median of a dataset. Given $n$ points $x_1, \ldots, x_n \in \mathbb{R}^d$, our goal is to find a point $\theta$ that minimizes the sum of Euclidean distances to these points while preserving privacy.

Unlike standard approaches like DP-GD that require strong a priori knowledge about data location within a ball of radius $R$ (with excess risk depending linearly on $R$), our algorithms provide excess error guarantees that scale with the **effective diameter** of the datapoints—the unknown radius containing the majority of the data.


## Contact

For questions and feedback:
- **Mahdi Haghifam** - [mhaghifam@ttic.edu](mailto:mhaghifam@ttic.edu)
- **GitHub Issues** - For bug reports and feature requests


---

**Note:** This is research code. While we have tested it thoroughly, it may contain bugs. Please report any issues you encounter.
