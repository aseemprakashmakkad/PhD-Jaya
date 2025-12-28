#!/usr/bin/env python3
"""
Auto-analysis script for the ScalesData CSV.
Produces:
 - scales_summary.csv  : per-column summary (type, missing, unique, etc.)
 - numeric_stats.csv   : numeric columns stats (count, mean, median, std, iqr, outliers)
 - scales_analysis_report.txt : human readable report

Usage: python3 analysis_scales.py /absolute/path/to/ScalesData.csv
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# By default analyze the Income-normalized file produced by `normalize_income.py`.
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/pranav/PhD-Jaya/InputData/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized.csv"

p = Path(CSV_PATH)
if not p.exists():
    print(f"CSV not found: {CSV_PATH}")
    sys.exit(2)

# Find header row by scanning for the line that starts with 'Name,' (case-sensitive) or contains 'Name,'
with p.open('r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

header_row_idx = None
for i, line in enumerate(lines[:50]):
    # look for a line that starts with Name, or contains 'Name,' as a column header
    if line.lstrip().startswith('Name,') or ',Name,' in line or line.strip().startswith('Name'):
        header_row_idx = i
        break
# fallback: try to detect a likely header by finding the line with many typical column names
if header_row_idx is None:
    for i, line in enumerate(lines[:80]):
        if 'Age' in line and 'Name' in line:
            header_row_idx = i
            break

if header_row_idx is None:
    # as last resort, assume header at line 4 (0-based index 4 -> 5th line)
    header_row_idx = 4

print(f"Detected header row index (0-based): {header_row_idx}")

# Read CSV using detected header
try:
    df = pd.read_csv(p, header=header_row_idx, encoding='utf-8', engine='python')
except Exception as e:
    print('Error reading CSV with pandas:', e)
    sys.exit(3)

# Basic shape
n_rows, n_cols = df.shape
print(f"Loaded dataframe with {n_rows} rows and {n_cols} columns")

# Clean column names: strip whitespace
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

# Per-column summary
summary = []
for col in df.columns:
    ser = df[col]
    n_missing = ser.isna().sum()
    n_unique = ser.nunique(dropna=True)
    dtype = str(ser.dtype)
    sample_values = ser.dropna().astype(str).head(3).tolist()
    summary.append({
        'column': col,
        'dtype': dtype,
        'n_missing': int(n_missing),
        'pct_missing': float(n_missing) / n_rows if n_rows>0 else np.nan,
        'n_unique': int(n_unique),
        'sample_values': ' | '.join(sample_values)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(p.parent / 'scales_summary.csv', index=False)

# Numeric stats & outlier detection
numeric = df.select_dtypes(include=[np.number])
num_stats = []
for col in numeric.columns:
    ser = numeric[col].dropna()
    if ser.empty:
        continue
    q1 = ser.quantile(0.25)
    q3 = ser.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ser[(ser < lower) | (ser > upper)]
    num_stats.append({
        'column': col,
        'count': int(ser.count()),
        'mean': float(ser.mean()),
        'median': float(ser.median()),
        'std': float(ser.std()),
        'min': float(ser.min()),
        'max': float(ser.max()),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'n_outliers': int(outliers.count()),
        'pct_outliers': float(outliers.count()) / ser.count() if ser.count()>0 else 0.0
    })

num_stats_df = pd.DataFrame(num_stats)
num_stats_df.to_csv(p.parent / 'numeric_stats.csv', index=False)

# Categorical quick summary (top values)
cat = df.select_dtypes(include=['object', 'category'])
cat_summary = []
for col in cat.columns:
    ser = cat[col].astype(str)
    top = ser.value_counts(dropna=True).head(5).to_dict()
    cat_summary.append({'column': col, 'n_unique': int(ser.nunique(dropna=True)), 'top_values': str(top)})
cat_summary_df = pd.DataFrame(cat_summary)
cat_summary_df.to_csv(p.parent / 'categorical_summary.csv', index=False)

# Compose a short report
report_lines = []
report_lines.append(f"File analyzed: {p}\nRows: {n_rows}  Columns: {n_cols}\n")
report_lines.append("Columns with >20% missing values:\n")
high_missing = summary_df[summary_df['pct_missing'] > 0.2]
if not high_missing.empty:
    for _, r in high_missing.iterrows():
        report_lines.append(f" - {r['column']}: {r['n_missing']} missing ({r['pct_missing']:.1%})\n")
else:
    report_lines.append(" - None\n")

report_lines.append('\nNumeric columns summary (top issues):\n')
if not num_stats_df.empty:
    # show columns with >1% outliers or extremely skewed (mean/median diff)
    flagged = []
    for _, r in num_stats_df.iterrows():
        if r['pct_outliers'] > 0.01 or abs(r['mean'] - r['median']) > max(1.5 * r['std'], 1e-6):
            flagged.append(r)
    if flagged:
        for r in flagged:
            report_lines.append(f" - {r['column']}: mean={r['mean']:.2f}, median={r['median']:.2f}, std={r['std']:.2f}, outliers={r['n_outliers']} ({r['pct_outliers']:.1%})\n")
    else:
        report_lines.append(" - No major numeric issues detected by simple heuristics\n")
else:
    report_lines.append(" - No numeric columns detected\n")

report_lines.append('\nTop recommended next steps:\n')
report_lines.append(' 1) Normalize/clean columns with number-like text (e.g., income ranges containing commas) if you need numeric analysis.\n')
report_lines.append(' 2) For columns with >20% missingness consider drop or imputation depending on importance.\n')
report_lines.append(' 3) Visualize numeric distributions and boxplots for columns flagged above.\n')
report_lines.append(' 4) If multi-row header semantics are important (e.g. groups), consider manually collapsing multirow headers into single descriptive names.\n')

# Write report
report_path = p.parent / 'scales_analysis_report.txt'
with report_path.open('w', encoding='utf-8') as fh:
    # write the report lines and ensure the file ends with a newline
    fh.writelines('\n'.join(report_lines))
    fh.write('\n')

print('\n'.join(report_lines))
print('\nSaved files:')
print(' -', p.parent / 'scales_summary.csv')
print(' -', p.parent / 'numeric_stats.csv')
print(' -', p.parent / 'categorical_summary.csv')
print(' -', report_path)

# Also write a lightweight CSV that is intended as the canonical input for plotting
# (so that `produce_report.py` can consume a single known file).
analysis_ready = p.parent / (p.stem + '-analysis-ready' + p.suffix)
try:
    df.to_csv(analysis_ready, index=False, encoding='utf-8')
    print(' - analysis-ready CSV written to', analysis_ready)
except Exception as _e:
    print('Could not write analysis-ready CSV:', _e)

# Exit cleanly
sys.exit(0)
