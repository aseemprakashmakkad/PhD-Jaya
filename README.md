## PhD-Jaya — Scales Data analysis

This repository contains scripts and data for exploratory analysis of the Scales dataset.

Quick overview
- Input data: `InputData/` (CSV/ODS files). The main working file used in scripts is:
  - `InputData/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized.csv`
- Scripts: `scripts/analysis_scales.py`, `scripts/normalize_income.py`, `scripts/produce_report.py`
- Outputs:
  - Summary CSVs: `scales_summary.csv`, `numeric_stats.csv`, `categorical_summary.csv`
  - Plots (local, not committed): `outputs/plots/` (PNG files)
  - Consolidated report PDF (tracked): `scales_plots_report.pdf`

Dependencies
- Python 3.8+ (or later)
- Required packages (install into a virtualenv):

```bash
pip install pandas numpy matplotlib seaborn
```

How to run
- Analysis (summary statistics):

```bash
cd /home/pranav/PhD-Jaya
python3 scripts/analysis_scales.py InputData/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized.csv
```

- Normalize income (creates `-IncomeNormalized.csv` in the same folder):

```bash
python3 scripts/normalize_income.py InputData/20251214-ScalesData-Combined_ver0.7-Cleaned.csv
```

- Produce plots and consolidated PDF report (writes PNGs to `outputs/plots/` and `scales_plots_report.pdf`):

```bash
python3 scripts/produce_report.py InputData/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized.csv
```

Notes & recommendations
- Scripts accept an explicit path to a CSV file. If you run them without arguments they will use the repository's default `InputData/` filenames (created during the analysis session).
- The `outputs/plots/` folder is `.gitignore`d so PNGs are kept local. The consolidated PDF report `scales_plots_report.pdf` is tracked in git.
- If you'd like scripts to auto-detect the newest `ScalesData` file in `InputData/` instead of using a hard-coded filename, I can update them to do so.

Contact / reproducibility
- To reproduce on another machine: clone the repo, create a Python virtual environment, install the dependencies above, and run the scripts as shown.

License
- Contents are project-specific; add an appropriate LICENSE file if you plan to publish.
