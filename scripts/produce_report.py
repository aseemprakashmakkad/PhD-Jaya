#!/usr/bin/env python3
"""
Produce EDA plots and a PDF report for the ScalesData CSV.
Generates:
 - outputs/plots/<col>_hist.png and <col>_box.png for numeric columns
 - outputs/plots/<col>_bar.png for categorical columns
 - scales_plots_report.pdf (combined report)

Usage: python3 produce_report.py /path/to/CSV
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/pranav/PhD-Jaya/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized.csv')
OUT_DIR = CSV.parent / 'outputs' / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUT = CSV.parent / 'scales_plots_report.pdf'

print('Reading', CSV)
df = pd.read_csv(CSV, engine='python', encoding='utf-8')
print('Shape:', df.shape)

# detect numeric and categorical
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
cat = df.select_dtypes(include=['object','category']).columns.tolist()

# Flag numeric columns with IQR-based outliers or mean/median divergence
flagged_numeric = []
for col in numeric:
    ser = df[col].dropna()
    if ser.empty:
        continue
    q1 = ser.quantile(0.25)
    q3 = ser.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ser[(ser < lower) | (ser > upper)]
    mean = ser.mean()
    median = ser.median()
    std = ser.std()
    pct_out = outliers.count() / ser.count() if ser.count() > 0 else 0
    if pct_out > 0.01 or abs(mean - median) > max(1.5 * std, 1e-9):
        flagged_numeric.append(col)

# For convenience, also include top N numeric columns if flagged list small
if len(flagged_numeric) < 8:
    # add top variance numeric columns
    variances = df[numeric].var().sort_values(ascending=False)
    for c in variances.index[:min(8, len(variances))]:
        if c not in flagged_numeric:
            flagged_numeric.append(c)

print('Flagged numeric columns:', flagged_numeric)

# Prepare PDF
pp = PdfPages(str(PDF_OUT))
# Title page
plt.figure(figsize=(11.7,8.3))
plt.axis('off')
plt.text(0.5, 0.7, 'Scales Data EDA Report', ha='center', va='center', fontsize=20)
plt.text(0.5, 0.6, f'File: {CSV.name}', ha='center', va='center', fontsize=10)
plt.text(0.5, 0.55, f'Rows: {len(df)}  Columns: {len(df.columns)}', ha='center', va='center', fontsize=10)
plt.text(0.05, 0.3, 'Notes:\n - Numeric flagged by simple IQR/outlier heuristics\n - Categorical plots limited to top categories', fontsize=9)
pp.savefig()
plt.close()

# helper
def sanitize(s):
    return ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in s).strip().replace(' ', '_')

# Numeric plots
# Numeric plots
for col in flagged_numeric:
    ser = df[col].dropna()
    if ser.empty:
        continue
    # Histogram with descriptive text below
    import textwrap
    n = int(ser.count())
    mean = float(ser.mean())
    median = float(ser.median())
    std = float(ser.std())
    mn = float(ser.min())
    mx = float(ser.max())
    q1 = float(ser.quantile(0.25))
    q3 = float(ser.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ser[(ser < lower) | (ser > upper)]
    n_out = int(outliers.count())
    pct_out = n_out / n if n>0 else 0
    skew_note = ''
    if mean - median > 0.5 * std:
        skew_note = 'Distribution appears right-skewed.'
    elif median - mean > 0.5 * std:
        skew_note = 'Distribution appears left-skewed.'
    # Compose description
    desc = (f'Count={n}; mean={mean:.2f}; median={median:.2f}; std={std:.2f}; '
            f'min={mn:.2f}; max={mx:.2f}; outliers={n_out} ({pct_out:.1%}). {skew_note}')

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(ser, kde=True, ax=ax)
    ax.set_title(f'Histogram: {col}')
    ax.set_xlabel(col)
    # add description as wrapped text below the plot area
    wrapped = textwrap.fill(desc, 160)
    fig.text(0.01, 0.02, wrapped, ha='left', va='bottom', fontsize=9)
    plt.tight_layout(rect=[0,0.05,1,1])
    fname_hist = OUT_DIR / f'{sanitize(col)}_hist.png'
    fig.savefig(fname_hist)
    pp.savefig()
    plt.close(fig)

    # Boxplot with descriptive text
    fig, ax = plt.subplots(figsize=(8,3))
    sns.boxplot(x=ser, orient='h', ax=ax)
    ax.set_title(f'Boxplot: {col}')
    wrapped_box = textwrap.fill('IQR-based outliers count: {} ({:.1%}). {}'.format(n_out, pct_out, skew_note), 160)
    fig.text(0.01, 0.02, wrapped_box, ha='left', va='bottom', fontsize=9)
    plt.tight_layout(rect=[0,0.05,1,1])
    fname_box = OUT_DIR / f'{sanitize(col)}_box.png'
    fig.savefig(fname_box)
    pp.savefig()
    plt.close(fig)

# Categorical plots: choose top categorical columns with manageable cardinality
cat_to_plot = []
for col in cat:
    nunique = df[col].nunique(dropna=True)
    if nunique <= 30 and nunique > 1:
        cat_to_plot.append(col)
# limit the number
cat_to_plot = cat_to_plot[:20]
print('Categorical columns plotted:', cat_to_plot)

for col in cat_to_plot:
    ser = df[col].fillna('<<MISSING>>')
    vc = ser.value_counts().head(15)
    plt.figure(figsize=(8,4))
    sns.barplot(y=vc.index.astype(str), x=vc.values, palette='viridis')
    plt.title(f'Value counts: {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    fname_bar = OUT_DIR / f'{sanitize(col)}_bar.png'
    plt.savefig(fname_bar)
    pp.savefig()
    plt.close()

pp.close()
print('Saved PDF report to', PDF_OUT)
print('Saved individual plots to', OUT_DIR)

# helper

def sanitize(s):
    return ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in s).strip().replace(' ', '_')

# Write small index of files
with open(CSV.parent / 'outputs' / 'plots' / 'index.txt', 'w', encoding='utf-8') as fh:
    for p in sorted(OUT_DIR.iterdir()):
        fh.write(str(p.name) + '\n')

print('Done.')
