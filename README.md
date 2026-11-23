# Travel-time uncertainty in neural routing policies (stochastic CVRPTW)

Code, run logs and figure scripts for the MSc thesis

> **How Should Travel-Time Uncertainty Enter a Neural Routing Policy? A Controlled Study on the Stochastic CVRPTW**
> Aya Boughanmi, ISG Tunis / LGI2A, Université d'Artois, 2026.

The thesis version of the code is the tag [`msc-thesis`](https://github.com/AyaBoughanmi02/msc-stochastic-cvrptw/tree/msc-thesis).
`main` continues with the work for a conference paper on the training reward.

## What is in this repository

We study a capacitated vehicle routing problem with time windows in which travel
times are random and depend on the time of day. A POMO-trained policy (E-GAT encoder
and pointer decoder, after Lei et al., 2022) is given information about travel-time
uncertainty in different ways (as an input feature, as a gate in the encoder, or only
through the training reward), and each variant is compared to a model that sees only
the mean travel times. We also compare three training rewards (analytic, sampled and
hybrid) and several rules for choosing a route before the day starts, and we compare
the policies with PyVRP baselines, including a rolling-horizon one.

We show that the best-of-N selection standard in deterministic neural
routing, applied once the day is known, understates costs by up to 34% and reverses
the ranking of the models. With routes chosen in advance, the hybrid reward lowers the
cost of a mean-only model by about 3%, after which no statistically detectable difference
remains between the ways the variance enters the network. In our setting, the resulting
policy is cheaper than a deterministic re-planner (PyVRP) in four of six comparisons and
never significantly more expensive.

## Install

Tested with Python 3.11.9 on Windows 11 with one CUDA GPU (4 GB).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

`requirements.txt` pins `torch==2.13.0+cu126`; install the PyTorch build that matches
your CUDA version first if needed. PyVRP 0.14.0 is used for the classical baselines.

## Layout

```
src/
  data/        instance generators and the travel-time simulator
  envs/        CVRP, CVRPTW and stochastic CVRPTW environments
  models/      encoder, decoder, model variants and the factory that builds them
  train/       training utilities
  eval/        evaluation, selection rules, PyVRP baselines
scripts/       training, evaluation, analysis and figure scripts (run as modules)
experiments/
  thesis/      PowerShell queues that launched the thesis runs
  paper/       PowerShell queues for the paper experiments
runs/          logs and result files (.log, .json, .csv, .txt) of the reported runs
references/    reference costs for the CVRP reproduction
tests/         pytest suite
```

Checkpoints (`.pt`) and generated data are not in the repository.

## Reproducing the thesis results

All scripts are run from the repository root as modules, for example

```powershell
python -u -m scripts.make_fig_pitfall 2>&1 | Tee-Object -FilePath runs\make_fig_pitfall.log
```

| Thesis item | Script |
|---|---|
| Figure 3.1 (congestion ripple, variance vs distance) | `scripts/make_fig_simulator.py` |
| Table 3.2 (binding constraint) | `scripts/binding_table.py` |
| Figure 3.3 (one route on many days, three rewards) | `scripts/make_fig_route_days.py` |
| Figure 4.1 | `scripts/make_fig_pitfall.py` |
| Figure 4.2 | `scripts/make_fig_converge.py` |
| Figure 4.3 (validation curves) | `scripts/make_fig_valcurves.py` |
| Training | `scripts/train_stoch.py` (launched by `experiments/thesis/*.ps1`) |
| Multi-seed comparison | `scripts/seed_study.py` |
| PyVRP baselines | `scripts/pyvrp_reference.py`, `scripts/pyvrp_rolling.py` |
| Selection rules | `scripts/selection_study.py`, `scripts/selection_analysis.py` |
| Closed-form check (Appendix) | `scripts/check_closed_forms.py` |


## Model names in the code

Some names in the code and in the result files are older than the names in the thesis.

| Thesis | Code / result files |
|---|---|
| M0 | `M0` |
| M | `Dp` (D-prime) |
| R | `VGTR` (VGT-R) |


## Use of AI assistance

As declared in the thesis, an AI assistant (Opus 5.5, Anthropic) was used for help with
code and writing. The research questions, experimental design, runs and conclusions
are the author's responsibility.

## Citation

See `CITATION.cff`. MIT licence, see `LICENSE`.
