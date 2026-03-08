# Dynamic Cross-Subject BCI Alignment Research Project

**Research Topic**: Dynamic Conditional Alignment for Motor Imagery Brain-Computer Interfaces

**Date**: March 2026

**Status**: Literature Review Complete, Ready for Implementation

---

## 📁 Project Structure

```
research-dynamic-bci-alignment-2026-03/
├── README.md                    # This file
├── literature-review.md         # Comprehensive literature review (50+ papers, 2021-2026)
├── research-proposal.md         # Detailed research proposal with methodology and timeline
└── references.bib               # BibTeX references (45+ entries)
```

---

## 🎯 Research Objective

Develop a **Dynamic Conditional Alignment** algorithm that addresses the **representation-behavior inconsistency problem** in cross-subject motor imagery EEG decoding by:

1. **Conditional Alignment**: Predicting alignment parameters based on current trial context
2. **Behavior-Guided Feedback**: Using online performance indicators to dynamically adjust alignment
3. **Closed-Loop Adaptation**: Creating a self-correcting system that adapts to within-session dynamics

---

## 📊 Key Findings from Literature Review

### Current State-of-the-Art

1. **Static Alignment Methods** (EA, RA, CORAL)
   - Strengths: Simple, computationally efficient
   - Limitations: Assume fixed domain relationships, no within-session adaptation

2. **Deep Domain Adaptation** (DANN, ADDA)
   - Strengths: Learn domain-invariant features
   - Limitations: Require large datasets, static after training

3. **Online Adaptation** (OTTA, EDAPT, T-TIME)
   - Strengths: Adapt during inference
   - Limitations: Use fixed adaptation rules, no context-awareness

4. **Meta-Learning** (MAML, Reptile)
   - Strengths: Enable few-shot adaptation
   - Limitations: High computational cost, require meta-training

### Identified Research Gap

**No existing method combines conditional alignment (context-aware parameter adjustment) with behavior-guided feedback (using prediction confidence and error patterns to dynamically update alignment strategies).**

---

## 🚀 Proposed Innovation

### Dynamic Conditional Alignment with Behavior-Guided Feedback (DCA-BGF)

**Core Components**:

1. **Conditional Alignment Network** g(x_t, c_t) → θ_t
   - Input: Current trial features + context (recent trial statistics)
   - Output: Alignment parameters (affine transformation or MLP parameters)

2. **Context c_t Design**:
   - Prediction entropy (uncertainty)
   - Class distribution shift
   - Confidence trend
   - Feature statistics (mean, std)

3. **Behavior-Guided Feedback**:
   - High uncertainty → strengthen alignment
   - Decreasing confidence → adapt more aggressively
   - Class distribution shift → recalibrate decision boundary

**Advantages**:
- ✅ Adapts to within-session dynamics (fatigue, attention shifts)
- ✅ Context-aware (not fixed rules like existing TTA methods)
- ✅ Lower computational cost than meta-learning
- ✅ Explicitly models representation-behavior inconsistency

---

## 📈 Experimental Plan

### Datasets
- **BCI Competition IV Dataset 2a**: 4-class MI, 9 subjects, 22 channels
- **BCI Competition IV Dataset 2b**: 2-class MI, 9 subjects, 3 channels

### Evaluation Protocol
- **Cross-Subject**: Leave-One-Subject-Out (LOSO)
- **Within-Session Dynamics**: Sliding window analysis
- **Representation-Behavior Analysis**: CKA, CCA, confusion matrix similarity

### Baselines
- Static: EA, RA, CORAL
- Deep DA: DANN, ADDA
- Online: OTTA, EDAPT, T-TIME
- Meta-Learning: MAML, Reptile

### Ablation Studies
1. Conditional alignment only (no behavior feedback)
2. Behavior-guided feedback only (no conditional alignment)
3. Context component analysis (H_t, D_t, C_t)
4. Alignment parameter form (affine vs. MLP)
5. Context window size (N ∈ {10, 20, 50, 100})

---

## 📅 Timeline (12 Months)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1**: Infrastructure Setup | Months 1-2 | Baseline results on BCI Competition IV |
| **Phase 2**: Method Development | Months 3-5 | Working DCA-BGF implementation |
| **Phase 3**: Experimental Validation | Months 6-8 | Comprehensive experimental results |
| **Phase 4**: Theoretical Analysis | Months 9-10 | Convergence proofs, generalization bounds |
| **Phase 5**: Paper Writing | Months 11-12 | Submitted paper to NeurIPS/ICML/ICLR |

---

## 🎓 Target Venues

- **Primary**: NeurIPS 2026 (May deadline), ICML 2026 (January deadline)
- **Secondary**: ICLR 2027 (September deadline)
- **Backup**: Journal of Neural Engineering, NeuroImage

---

## 📚 Literature Summary

### Papers Reviewed: 50+
### Time Period: 2021-2026
### Key Topics:
- Transfer learning & domain adaptation (10 papers)
- Riemannian geometry methods (8 papers)
- Online adaptation & calibration-free BCI (7 papers)
- Meta-learning (5 papers)
- Disentangled representation learning (6 papers)
- Behavior-guided learning (4 papers)
- Continual learning (5 papers)
- Conditional alignment (5 papers)

### Most Relevant Papers:

1. **[Revisiting Euclidean Alignment for Transfer Learning in EEG-Based BCIs](https://arxiv.org/html/2502.09203v2)** (2025)
   - Re-examines EA variants, proposes improved EA with regularization

2. **[Geometry-Aware Deep Congruence Networks for Cross-Subject MI](https://arxiv.org/html/2511.18940v1)** (2025)
   - Deep learning on SPD manifolds for cross-subject MI

3. **[Towards Calibration-Free BCIs with Continual Online Adaptation](https://arxiv.org/html/2508.10474)** (2025)
   - EDAPT framework for continual adaptation without calibration

4. **[The Decoupling Hypothesis: Subject-Invariant EEG Representation Learning](https://iclr-blogposts.github.io/2026/blog/2026/subject-invariant-eeg/)** (2026)
   - Disentangles subject-specific artifacts and temporal trends

5. **[Neural Decoding through Multi-subject Class-conditional Hyperalignment](https://openreview.net/forum?id=AQEnNqRuUU)** (2024)
   - MuSCH: class-conditional alignment for multi-subject decoding

---

## 💡 Next Steps

1. **Immediate**:
   - Set up development environment (PyTorch, MNE-Python, MOABB)
   - Download BCI Competition IV datasets
   - Implement baseline methods (EA, RA, CORAL)

2. **Short-term** (Weeks 1-4):
   - Reproduce baseline results
   - Implement conditional alignment network
   - Design context c_t and behavior-guided feedback

3. **Medium-term** (Months 2-5):
   - Full DCA-BGF implementation
   - Hyperparameter tuning
   - Initial experimental validation

4. **Long-term** (Months 6-12):
   - Comprehensive experiments
   - Theoretical analysis
   - Paper writing and submission

---

## 📖 How to Use This Repository

### Reading Order:
1. **Start here**: `README.md` (overview and quick start)
2. **Deep dive**: `literature-review.md` (comprehensive review of 50+ papers)
3. **Methodology**: `research-proposal.md` (detailed research plan)
4. **References**: `references.bib` (BibTeX entries for all cited papers)

### For Implementation:
- Use `literature-review.md` Section 3-9 for baseline method details
- Use `research-proposal.md` Section 2-3 for DCA-BGF algorithm specification
- Use `research-proposal.md` Section 3 for experimental protocol

### For Paper Writing:
- Use `literature-review.md` Section 10 for gap analysis
- Use `research-proposal.md` Section 4 for contributions
- Use `references.bib` for citations

---

## 🔗 Useful Resources

### Datasets:
- [BCI Competition IV](https://bbci.de/competition/iv/)
- [MOABB (Mother of All BCI Benchmarks)](https://github.com/NeuroTechX/moabb)

### Code Libraries:
- [MNE-Python](https://mne.tools/) - EEG processing
- [PyRiemann](https://pyriemann.readthedocs.io/) - Riemannian geometry
- [MOABB](https://github.com/NeuroTechX/moabb) - BCI benchmarking

### Related Repositories:
- [EEGNet](https://github.com/vlawhern/arl-eegmodels) - Deep learning for EEG
- [BrainDecode](https://github.com/braindecode/braindecode) - Deep learning for EEG

---

## 📧 Contact

For questions or collaboration inquiries, please contact [Your Email].

---

## 📄 License

This research project is for academic purposes. Please cite appropriately if you use any materials from this repository.

---

**Last Updated**: March 5, 2026
