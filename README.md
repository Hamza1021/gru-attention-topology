# GRU Cued Change Detection and Hidden-State Topology

## 1. Project summary

**Scientific question:** Does supervised training on a cued visual change-detection task produce behaviorally useful cue representations and cue-dependent structure in the recurrent hidden state? **Hypothesis:** Informative cues should improve detection performance, remain decodable after cue offset, and induce measurable changes in hidden-state or hidden-unit correlation topology. A one-layer, two-head gated recurrent unit (GRU) is trained on simulated noisy left and right visual streams and evaluated under true, neutral, flipped, and shuffled cue conditions. Behavioral metrics, leakage-safe cue decoding, persistent homology, Rouse-like hidden-unit correlation analyses, and matched null controls are compared across five independently trained random seeds.

## 2. Repository structure

```text
neuro_gru_cued_change_detection/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
└── gru_cued_change_detection_topology.ipynb
```

The notebook contains the complete workflow: task simulation, supervised GRU training, response-threshold calibration, behavioral evaluation, cue decoding, persistent-homology analyses, null controls, aggregation across seeds, figure generation, reporting, and export of a ZIP archive.

A full run creates a timestamped directory in the current working directory:

```text
RUN_THIS_gru_cue_specific_topology_results_<timestamp>/
├── config.json
├── summary_report.md
├── csv/                         # Aggregated seed-level and summary tables
├── figures/                     # Main presentation figures
├── seed_00/ ... seed_04/        # Per-seed models, metrics, plots, and topology
└── topology/                    # Reserved top-level topology directory
```

Each `seed_XX/` directory contains the final model checkpoint, training history, threshold sweep, trial-level behavior, decoder outputs, and per-seed topological analyses. Generated result directories and model checkpoints are excluded by `.gitignore` because they can be regenerated and may be too large for a standard GitHub repository.

## 3. How to run

### Python version

The notebook metadata records **Python 3.10.20**. Python 3.10 is therefore the recommended environment.

### Option A: `venv` and `requirements.txt`

macOS/Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `gru_cued_change_detection_topology.ipynb`, select the environment kernel, restart the kernel, and run the notebook from top to bottom.

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate neuro-gru-cue-topology
jupyter lab
```

### Main dependencies

The reproducible package versions are listed in `requirements.txt` and `environment.yml`. The principal libraries are:

- PyTorch for the GRU model and supervised training;
- NumPy and pandas for simulation and tabular results;
- scikit-learn for scaling, PCA, logistic-regression decoding, and grouped train/test splitting;
- Ripser and Persim for persistent homology and persistence diagrams;
- Matplotlib for figures;
- JupyterLab and IPython kernel support for notebook execution.

### Full run versus debugging run

The configuration cell contains:

```python
QUICK_RUN = False
RUN_TOPOLOGY = True
```

Use `QUICK_RUN = False` and `RUN_TOPOLOGY = True` for the final five-seed experiment. Set `QUICK_RUN = True` only for a short installation and code-path check; quick-run outputs are not final scientific results.

### Reproducing the results

The full notebook performs the following sequence:

1. Simulates 36-timestep trials with a transient cue, two noisy visual streams, pre-change distractors, and a limited post-change response window.
2. Trains a one-layer GRU with 64 hidden units and separate response and side-choice heads.
3. Runs five independent seeds (`0`–`4`), each for 220 epochs, 25 batches per epoch, and a batch size of 256.
4. Calibrates the response threshold for each seed on 1,600 validation trials using  
   `accuracy - 2 × false_alarm_rate - 0.5 × lapse_rate`.
5. Evaluates each trained model on 2,500 matched trials under true, neutral, flipped, and shuffled cue modes.
6. Extracts pre-change hidden states from 1,200 additional trials for cue decoding and topological analyses.
7. Computes hidden-state point-cloud topology and Rouse-like hidden-unit correlation topology, including Gaussian, shuffled, and other null controls.
8. Aggregates means and standard errors across trained seeds, saves CSV tables and PNG figures, writes `summary_report.md`, and archives the complete output directory.
9. Runs the final standalone leakage-safe decoder cell, which uses fresh trials, a trial-grouped 70/30 split, training-only scaling, and trial-level label permutations without retraining the GRU.

The final standalone decoder cell can be rerun by itself after a completed training run, provided the timestamped results directory still contains `seed_XX/final_model.pt`.

### Generating figures

All figures are generated automatically when the notebook is executed. Main aggregate figures are written to:

```text
RUN_THIS_gru_cue_specific_topology_results_<timestamp>/figures/
```

Per-seed training, threshold-calibration, persistence-diagram, Betti-curve, and correlation-matrix figures are written under the corresponding `seed_XX/` directories.

### Expected runtime

Runtime depends on hardware and on the cost of the persistent-homology calculations.

- **Quick run:** approximately 2–10 minutes on a modern laptop CPU.
- **Full five-seed run:** approximately 25 min on a modern laptop CPU.

These are approximate ranges. Processor speed, GPU availability, Python environment, and system load can change the runtime substantially.

## 4. Results and interpretation

The notebook reports task accuracy, balanced accuracy, valid-versus-invalid performance, false-alarm, lapse, and late-response rates for each cue condition. It also tests whether cue identity is linearly decodable from post-cue, pre-change hidden states on unseen trials and compares observed persistent-homology summaries with structure-destroying null controls.

The main results should be interpreted at the level supported by the corresponding control:

- Above-chance behavior establishes that the GRU learned the detection task.
- True-versus-neutral, flipped, and shuffled comparisons test the behavioral usefulness of the cue.
- Trial-grouped decoding establishes that cue identity is represented in hidden states after cue offset, but decoding alone is not proof of causal attentional use.
- Differences from shuffled or independent nulls establish nonrandom topology.
- Similar topology across cue modes should be reported as a null or weak cue-specific effect rather than as evidence of strong attentional reorganization.

The independently trained random seed is the inferential unit for aggregate means and standard errors.

## 5. Documentation of LLM Usage

ChatGPT (**GPT-5.6 Thinking**) was used to assist with implementation and debugging of notebook code. 