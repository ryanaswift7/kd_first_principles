# KDFP: Knowledge Distillation from First Principles

## Abstract

Knowledge distillation is an established technique for improving the capabilities of small, efficient student models by training them with the representations of larger, more capable teacher models. Much of the recent work in the distillation of large language models (LLMs) has focused on distilling abilities learned during post-training, such as instruction following, chain-of-thought reasoning, and tool usage. This has left a large research gap in general knowledge distillation for LLMs, which is essential for developing efficient and private systems suitable for deployment on edge devices. We take a first-principles approach, evaluating previous lessons from prior works and conducting new explorations to develop a distillation methodology suitable for modern LLMs. We present KDFP, a novel methodology for white-box general knowledge distillation in LLMs. We demonstrate that KDFP outperforms existing methods by 1.6\% -- 4.9\% across 9 benchmarks while increasing training efficiency by up to 99.1\% through ephemeral parameter reduction.

## Paper

[KDFP: A first-principles approach to knowledge distillation in large language models](https://arxiv.org/abs/2610.10854)

## Environment Setup

Create the pinned Conda environment and install this repository as a package:

```bash
conda env create -f environment.yaml
conda activate kdfp
pip install -e .
```

Run the commands below from the repository root.

## Reproducing Results

### Trimodal Divergences Figure

Generate the trimodal divergences figure (Figure 2) from the analytic distributions defined in
`kdfp/utils/make_jsd_fig.py`:

```bash
make-jsd-fig
```

The command writes the following PDF to the repository root:

```text
divergences_trimodal.pdf
```

### Exploration Experiments

Running the following script will reproduce the experiments we ran during our exploration (Appendix F)

```bash
bash scripts/exploration_exps.sh
```


### Main Experiments

This script will reproduce all of our main KDFP results from the paper (Section 7.1)

```bash
bash scripts/main_kdfp_results.sh
```

### Analyses

This will reproduce all of our analyses from Sections 7.2--7.4

```bash
bash scripts/analyses.sh
```

### TED Results

This script will reproduce all of the TED results for all of Section 7

```bash
bash scripts/all_ted.sh
```


### Result Compilation

After running the experiments, compile the results into labeled plots by passing a
timestamped Parquet results file to `plot-gen`:

```bash
plot-gen path/to/results_df_<timestamp>.parquet
```

KDFP writes these files to the suite score directory, while student evaluation
results are written to the model's `scores/` directory. The command creates a
`plots/` directory alongside the input file containing per-task scores, average
scores, and perplexity plots.


## Final Note

As a word of caution to anyone who may wish to use our experiment harness for themselves, you should know that it is ***not*** production-grade whatsoever. There was a lot of rapid development toward the end, which resulted in breaking some configuration execution paths. While these configurations and this harness will indeed reproduce our experiments, it will need to be refined if you intend to use it as the general, declarative experiment harness we hoped it might become one day. It contains a lot of capabilites which we built and leveraged at different points, but which we elected to break for the sake of simpler, more efficient code execution at different times in the development of KDFP.

**TL;DR** This experiment harness will reproduce our experiments, but if you want to use it for yourself, be warned that some of the exposed functionality is currently non-functional since it was not critical for the paper.


## Citation

```bibtex
@article{swift2026kdfp,
  title         = {KDFP: A first-principles approach to knowledge distillation in large language models},
  author        = {Swift, Ryan and Psounis, Konstantinos},
  journal       = {arXiv preprint arXiv:2610.10854},
  year          = {2026}
}
```
