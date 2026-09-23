# Multi-Omics AMR Prediction with Attention Fusion

> 🚧 **Status: In progress.** This project is in the planning and early development stage. No results yet; this README describes the design and roadmap.

Predicting antimicrobial resistance (AMR) in *E. coli* by combining whole-genome sequencing, pangenome gene content, and transcriptomics in a single explainable deep-learning model.

## Background

This project builds on:

> Ren et al. (2022). *Prediction of Antimicrobial Resistance Based on Whole-Genome Sequencing and Machine Learning.* Bioinformatics. [doi:10.1093/bioinformatics/btab681](https://doi.org/10.1093/bioinformatics/btab681) · [Original code](https://github.com/YunxiaoRen/ML-iAMR)

The original study trained Logistic Regression, SVM, Random Forest, and CNN models on *E. coli* WGS data to predict resistance to four antibiotics, and introduced **FCGR (Frequency Chaos Game Representation)** to encode genomes as 2D images for a CNN. Its models performed well within the training data but dropped on an independent dataset, a generalization gap attributed to population structure, plasmid diversity, and novel resistance mechanisms.

This project first replicates the WGS-only baseline, then asks whether adding more biological layers closes that gap.

## Research Question

**Does integrating genomic sequence (WGS), gene content (pangenome), and gene expression (transcriptomics) in an attention fusion model improve AMR prediction accuracy and biological interpretability compared with WGS alone?**

## Target Antibiotics

The four antibiotics were chosen because each one tests a different resistance mechanism.

| Antibiotic | Class | Primary resistance mechanism |
|---|---|---|
| Ciprofloxacin | Fluoroquinolone | Target mutation (*gyrA* / *parC* SNPs) |
| Cefotaxime | 3rd-gen cephalosporin | ESBL enzymatic inactivation |
| Ceftazidime | 3rd-gen cephalosporin | ESBL + AmpC beta-lactamases |
| Gentamicin | Aminoglycoside | Aminoglycoside-modifying enzymes |

## Approach

### Three omics layers

| Layer | What it captures | Representation | Model branch |
|---|---|---|---|
| **WGS** | SNPs, resistance gene sequences, sequence composition | 64×64 FCGR image | CNN |
| **Pangenome** | Presence/absence of core and accessory genes, including plasmid-borne resistance genes | Binary vector (~5,000 genes) | MLP |
| **Transcriptomics** | Gene expression under antibiotic stress, such as efflux pump upregulation | Normalized expression of top ~500 DEGs | MLP |

### Model architecture

```
  WGS (64×64 FCGR)    Pangenome (~5000)    Transcriptomics (~500)
        │                    │                      │
    CNN branch           MLP branch             MLP branch
        │                    │                      │
        └──────────── Attention fusion ─────────────┘
                             │
                      Classifier head
                             │
                Resistance probability (0–1)
```

The attention fusion layer learns a weight for each omics branch **per prediction**. SNP-driven resistance should lean on WGS, while efflux-driven resistance should lean on transcriptomics. These weights are themselves interpretable.

### Explainability

**SHAP** values will be used to produce:
- global feature importance, to check whether the model recovers known AMR genes and flags candidate novel markers;
- per-sample explanations of which features drive an individual prediction;
- omics-layer importance, to test whether transcriptomics adds real value.

### Ablation study

Seven model variants will be trained and compared:

1. WGS only (replicates the original paper)
2. Pangenome only
3. Transcriptomics only
4. WGS + Pangenome
5. WGS + Transcriptomics
6. Pangenome + Transcriptomics
7. All three (full model)

## Tech Stack

| Area | Tools |
|---|---|
| Languages | Python 3.9+, R, Bash |
| Deep learning | PyTorch |
| Machine learning | scikit-learn, imbalanced-learn (SMOTE), Optuna |
| Explainability | SHAP, matplotlib, seaborn |
| Genomics | BioPython, Prokka, Roary / Panaroo |
| Transcriptomics | HISAT2, SAMtools, featureCounts, DESeq2 |
| Data & compute | NumPy, pandas, conda, SLURM (Northeastern Discovery HPC) |

## Data Sources

- **[BV-BRC (PATRIC)](https://www.bv-brc.org)**: *E. coli* genomes with AMR phenotype labels
- **[NCBI GEO / SRA](https://www.ncbi.nlm.nih.gov/geo)**: *E. coli* RNA-seq under antibiotic stress

## Roadmap

- [ ] Environment setup (conda + dependencies)
- [ ] FCGR encoding: FASTA → 64×64 image
- [ ] PyTorch `Dataset` / `DataLoader`
- [ ] CNN branch
- [ ] Training and evaluation loop
- [ ] Reproduce WGS-only baseline
- [ ] Pangenome branch: Prokka → Roary/Panaroo → MLP
- [ ] RNA-seq pipeline: HISAT2 → featureCounts → DESeq2
- [ ] Attention fusion network
- [ ] SHAP analysis
- [ ] 7-model ablation study
- [ ] SLURM scripts for full HPC runs

## Planned Repository Structure

```
├── data/            # raw and processed data (not tracked)
├── src/
│   ├── encoding/    # FCGR and feature encoding
│   ├── models/      # CNN, MLP branches, fusion model
│   ├── train.py
│   └── explain.py   # SHAP analysis
├── rnaseq/          # HISAT2 → DESeq2 pipeline
├── slurm/           # HPC job scripts
├── notebooks/
└── environment.yml
```

## Author

**Mayank Gandhi**, MS Bioinformatics, Northeastern University
[LinkedIn](https://www.linkedin.com/in/mayankgandhi0713) · [GitHub](https://github.com/mayankgandhi13)
