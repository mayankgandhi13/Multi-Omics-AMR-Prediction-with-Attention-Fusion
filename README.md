# Multi-Omics AMR Prediction with Attention Fusion

> 🚧 **Status: Phase 1 (WGS baseline) code is in and tested locally.** Full runs on Northeastern's Explorer HPC are next. No reported results yet.

Predicting antimicrobial resistance (AMR) in *E. coli* by combining whole-genome sequencing, pangenome gene content, and transcriptomics in a single explainable deep-learning model.

## The project in plain English

Bacteria that shrug off antibiotics are one of medicine's biggest headaches. Today, labs find out whether a drug will work by growing the bacteria with the drug and waiting, often for days. This project asks whether we can read the answer straight from the bacterium's DNA (and RNA) instead.

Think of each isolate as a suspect, and resistance as the crime. We have three kinds of evidence:

| Evidence | Detective analogy | What it catches |
|---|---|---|
| **WGS** (SNPs) | Fingerprints | Tiny spelling changes in drug targets, e.g. *gyrA* mutations behind ciprofloxacin resistance |
| **Pangenome** (gene presence/absence) | The suspect's toolbox | Whole resistance genes picked up from other bacteria, e.g. ESBL genes on plasmids |
| **Transcriptomics** (gene expression) | Phone records: who's active right now | Genes switched up under stress, e.g. efflux pumps working overtime |

A model that sees one kind of evidence can miss crimes the others would catch. The **attention fusion** model is a lead detective who decides, case by case, which evidence to trust most, and the attention weights show its reasoning.

## Background

This project builds on:

> Ren et al. (2022). *Prediction of Antimicrobial Resistance Based on Whole-Genome Sequencing and Machine Learning.* Bioinformatics. [doi:10.1093/bioinformatics/btab681](https://doi.org/10.1093/bioinformatics/btab681) · [Original code](https://github.com/YunxiaoRen/ML-iAMR)

The original study trained Logistic Regression, SVM, Random Forest, and CNN models on *E. coli* SNP data to predict resistance to four antibiotics, and used **FCGR (Frequency Chaos Game Representation)** to turn SNP profiles into 2D images for a CNN. Its models did well within the training data but dropped on an independent dataset, a generalization gap attributed to population structure, plasmid diversity, and novel resistance mechanisms.

This project first replicates the WGS-only baseline, then asks whether adding more biological layers closes that gap.

## Research Question

**Does integrating genomic sequence (WGS), gene content (pangenome), and gene expression (transcriptomics) in an attention fusion model improve AMR prediction accuracy and biological interpretability compared with WGS alone?**

## Target Antibiotics

The four antibiotics were chosen because each one tests a different resistance mechanism.

| Antibiotic | Class | Primary resistance mechanism |
|---|---|---|
| Ciprofloxacin (CIP) | Fluoroquinolone | Target mutation (*gyrA* / *parC* SNPs) |
| Cefotaxime (CTX) | 3rd-gen cephalosporin | ESBL enzymatic inactivation |
| Ceftazidime (CTZ) | 3rd-gen cephalosporin | ESBL + AmpC beta-lactamases |
| Gentamicin (GEN) | Aminoglycoside | Aminoglycoside-modifying enzymes |

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

## Design decisions (and why)

- **Lineage-aware splits.** *E. coli* comes in clonal families that share both SNPs and resistance. If siblings land on both sides of a train/test split, a model can score well by recognising the family rather than the biology. Every experiment runs twice: a `random` split (comparable to the paper) and a `lineage` split where families (isolates within 2% SNP distance) never cross folds. The gap between the two is the generalization gap.
- **The test fold stays sealed.** CNNs pick their best epoch on a validation slice carved from the training data, never on the test fold.
- **Class weights instead of SMOTE.** Rare resistant isolates are up-weighted during training rather than synthesised; fake genomes from SMOTE don't make biological sense.
- **Clinical error rates.** Alongside AUROC, AUPRC, MCC and F1, every model reports VME (resistant called susceptible, the dangerous mistake) and ME (susceptible called resistant).
- **FCGR caveat.** As in Ren et al., FCGR is drawn from each isolate's concatenated SNP alleles. Neighbouring letters can be far apart in the real genome, so the image is a fingerprint of the SNP profile rather than real k-mers.

## Quickstart

### On Explorer (Northeastern HPC)

```bash
git clone https://github.com/mayankgandhi13/Multi-Omics-AMR-Prediction-with-Attention-Fusion.git
cd Multi-Omics-AMR-Prediction-with-Attention-Fusion

# run all of Phase 1: env check -> download -> prepare -> baselines + CNNs (GPU) -> summary
bash slurm/run_all.sh
squeue -u $USER
```

The first run also builds the conda env (~10 min); later runs just check it. Paths live in [`slurm/common.sh`](slurm/common.sh): the env and data both sit on `/scratch/$USER`, because `/home` is far too slow for an env's thousands of small files. If a `/scratch` purge wipes them, the next `run_all.sh` rebuilds both. Partition names (`short`, `gpu`) are in each `.sbatch` header; check `sinfo -s` if yours differ.

### On a laptop

```bash
conda env create -f environment.yml && conda activate amr
bash scripts/get_data.sh
python -m src.prepare                                     # ~20 s
python -m src.baselines --antibiotic CIP --split lineage
python -m src.train_cnn --antibiotic CIP --split lineage --input fcgr
python -m src.summarize
```

## How the code flows

```
scripts/get_data.sh   download the Giessen SNP matrix (809 isolates)
        │
src/prepare.py        CSV -> data/processed/dataset.npz (SNPs, labels, families, FCGR)
        │
        ├── src/baselines.py   LR / SVM / RF  x  label / one-hot   x  random / lineage
        └── src/train_cnn.py   1D CNN (SNPs) / 2D CNN (FCGR)       x  random / lineage
        │
src/summarize.py      results/summary.csv: mean ± std per antibiotic x split x model
```

## Repository Structure

```
├── config.yaml          # every knob: antibiotics, folds, seed, FCGR k, CNN settings
├── environment.yml
├── scripts/
│   └── get_data.sh      # data download
├── src/
│   ├── data.py          # loading and caching
│   ├── encoding.py      # label, one-hot, FCGR
│   ├── splits.py        # SNP distances, lineage families, CV folds
│   ├── metrics.py       # AUROC, AUPRC, MCC, F1, VME, ME
│   ├── models.py        # 1D CNN (paper replication), 2D FCGR CNN (future WGS branch)
│   ├── prepare.py       # step 1
│   ├── baselines.py     # step 2
│   ├── train_cnn.py     # step 3
│   └── summarize.py     # step 4
├── slurm/               # Explorer job scripts (setup, prepare, baselines, cnn, summary, run_all)
├── data/                # raw and processed data (not tracked)
└── results/             # per-fold metrics, predictions, summary.csv
```

## Tech Stack

| Area | Tools |
|---|---|
| Languages | Python 3.11, R, Bash |
| Deep learning | PyTorch |
| Machine learning | scikit-learn, Optuna |
| Explainability | SHAP, matplotlib, seaborn |
| Genomics | BioPython, Bakta / Prokka, Panaroo |
| Transcriptomics | Salmon / Bowtie2, featureCounts, DESeq2 |
| Data & compute | NumPy, pandas, conda, SLURM (Northeastern Explorer HPC) |

## Data Sources

- **[Giessen dataset](https://github.com/YunxiaoRen/ML-iAMR)** (Ren et al. 2022): 809 *E. coli* isolates, SNP matrix + R/S labels for CIP, CTX, CTZ, GEN. Used for Phase 1.
- **[BV-BRC (PATRIC)](https://www.bv-brc.org)**: *E. coli* genomes with laboratory-measured AMR phenotypes
- **[NCBI GEO / SRA](https://www.ncbi.nlm.nih.gov/geo)**: RNA-seq data

## Roadmap

- [x] Environment setup (conda + Explorer SLURM scripts)
- [x] Data download and caching
- [x] Lineage-aware cross-validation splits
- [x] Encodings: label, one-hot, FCGR (SNP profile → 64×64 image)
- [x] LR / SVM / RF baselines
- [x] CNNs: 1D replication of the paper + 2D FCGR CNN
- [ ] Run Phase 1 on Explorer and compare with Ren et al.
- [ ] Rule-based baseline (AMRFinderPlus / ResFinder)
- [ ] Pangenome branch: Bakta → Panaroo → MLP
- [ ] Transcriptomics branch (needs isolates with paired genome + RNA-seq + phenotype)
- [ ] Attention fusion network
- [ ] SHAP analysis
- [ ] 7-model ablation study

## Author

**Mayank Gandhi**, MS Bioinformatics, Northeastern University
[LinkedIn](https://www.linkedin.com/in/mayankgandhi0713) · [GitHub](https://github.com/mayankgandhi13)
