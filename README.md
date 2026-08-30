<div align="center">

# WARBERT

### A Hierarchical BERT-Based Model for Web API Recommendation

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-00629B.svg)](https://doi.org/10.1109/TSC.2026.3688576)
[![arXiv](https://img.shields.io/badge/arXiv-2509.23175-b31b1b.svg)](https://arxiv.org/abs/2509.23175)
[![Venue](https://img.shields.io/badge/IEEE%20TSC-2026-00629B.svg)](https://www.computer.org/csdl/journal/sc)

**Zishuo Xu &middot; Yuhong Gu &middot; Dezhong Yao**

Official source-code release for **WARBERT: A Hierarchical BERT-Based Model for Web API Recommendation**, published in *IEEE Transactions on Services Computing*, vol. 19, no. 3, pp. 2591-2604, 2026.

</div>

## Overview

WARBERT is a hierarchical framework for recommending Web APIs from natural-language mashup requirements. It first retrieves a compact candidate set and then performs fine-grained semantic matching, combining the efficiency of multi-label recommendation with the accuracy of pairwise reranking.

## Method

- **WARBERT(R)** performs candidate filtering as a multi-label recommendation task. It uses dual-component feature fusion and optional mashup-category supervision.
- **WARBERT(M)** jointly encodes mashup and API descriptions, applies attention-based comparison, and reranks the retrieved candidates.
- **Score fusion** combines the two stages to produce the final ranking.

```text
Mashup requirement
        |
        v
WARBERT(R): candidate filtering
        |
        v
WARBERT(M): semantic matching
        |
        v
Score fusion -> ranked Web APIs
```

## Code Release

| File | Purpose |
| --- | --- |
| `warbert_r.py` | Training and evaluation driver for candidate filtering, including category-aware auxiliary supervision. |
| `warbert_m.py` | Training and evaluation driver for pairwise matching, negative sampling, and WARBERT(R)-guided hard-negative sampling. |

> **Reproducibility note.** This repository is a source snapshot of the two experiment drivers, not a standalone runnable package. The referenced `config.py`, `model.py`, `utils/metrics.py`, preprocessing code, processed dataset, and checkpoints are not included in the current release.

To reuse the scripts, provide the missing modules with the interfaces referenced by the imports and prepare the data objects described in the paper. The scripts expect a preprocessed `processed_data.pkl`; the preprocessing command mentioned in the source refers to a `data_preprocessor.py` file that is not part of this repository.

## Dataset

Experiments use the ProgrammableWeb mashup-API dataset after the preprocessing protocol described in the paper.

| Statistic | Value |
| --- | ---: |
| Mashups | 8,217 |
| Web APIs | 1,647 |
| Categories | 499 |
| Average APIs per mashup | 2.091 |

The dataset and preprocessing artifacts are not distributed in this repository.

## Citation

If you find this work useful, please cite the published article:

```bibtex
@article{xu2026warbert,
  title   = {WARBERT: A Hierarchical BERT-Based Model for Web API Recommendation},
  author  = {Xu, Zishuo and Gu, Yuhong and Yao, Dezhong},
  journal = {IEEE Transactions on Services Computing},
  volume  = {19},
  number  = {3},
  pages   = {2591--2604},
  year    = {2026},
  doi     = {10.1109/TSC.2026.3688576},
  url     = {https://doi.org/10.1109/TSC.2026.3688576}
}
```

## Contact

For questions or suggestions, please open an issue in this repository.
