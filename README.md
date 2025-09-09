# Retrieval-Augmented NeRAF (RAG-NVAS)

This repository extends [NeRAF: Neural Radiance and Acoustic Fields](https://github.com/AmandineBtto/NeRAF) by Bertolo *et al.* (NeurIPS 2023).  
While the original NeRAF introduced a neural field framework for **novel view acoustic synthesis (NVAS)**, this project augments NeRAF with a **retrieval-guided generation module** that injects acoustically diverse references to improve decay-related metrics while preserving spectral fidelity.

---

## Features
- Retrieval module (Siamese MLP with custom acoustic metrics: T60, C50, EDC, SPL, etc.)
- Cross-attention and FiLM-based fusion of retrieved references with baseline NeRAF
- Residual mask head for late-reverberation refinement
- Evaluation tools for spectral and decay metrics

---

