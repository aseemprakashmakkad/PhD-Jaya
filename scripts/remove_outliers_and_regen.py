#!/usr/bin/env python3
"""
Remove outliers from an Income-normalized CSV and regenerate plots + PDF into a new folder.

Usage: python3 remove_outliers_and_regen.py /path/to/IncomeNormalized.csv

Default input (if none provided):
 InputData/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized.csv

Behavior:
 - Detect numeric columns and mark as outliers any row where any numeric value
   is outside the column's IQR-based fences (q1-1.5*IQR, q3+1.5*IQR).
 - Remove those rows and save a new CSV alongside the input named
   <stem>-NoOutliers.csv
 - Redraw one page per variable (numeric: hist+box; categorical: bar) into a new
   plots folder under the input's parent called `outputs_nooutliers/plots` and
   write a PDF named `scales_plots_report_nooutliers.pdf`.
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
import textwrap


def sanitize(s):
    return ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in str(s)).strip().replace(' ', '_')


def remove_outliers(df, numeric_cols, min_outlier_cols=2, iqr_multiplier=1.5):
    """
    Remove rows that are outliers in at least `min_outlier_cols` numeric columns.

    - numeric_cols: list of numeric column names
    - min_outlier_cols: minimum number of columns that must be flagged as outlier for the row to be removed
    - iqr_multiplier: multiplier for the IQR fences (default 1.5). Increasing this makes outlier detection more permissive.

    Returns (df_kept, keep_mask) where keep_mask is a boolean Series marking rows kept.
    """
    # compute fences per column
    fences = {}
    for col in numeric_cols:
        ser = df[col].dropna()
        if ser.empty:
            continue
        q1 = ser.quantile(0.25)
        q3 = ser.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        fences[col] = (lower, upper)

    # count outlier flags per row
    outlier_counts = pd.Series(0, index=df.index)
    for col, (lower, upper) in fences.items():
        # True where value is an outlier (not NaN and outside fences)
        is_out = (~df[col].isna()) & ((df[col] < lower) | (df[col] > upper))
        outlier_counts += is_out.astype(int)

    # keep rows that have fewer than min_outlier_cols outlier flags
    keep = outlier_counts < int(max(1, min_outlier_cols))
    # return copy of kept rows and the boolean mask
    return df[keep].copy(), keep


def plot_all_columns(df, out_plots_dir, pdf_out):
    out_plots_dir.mkdir(parents=True, exist_ok=True)
    pp = PdfPages(str(pdf_out))
    cols = list(df.columns)
    for col in cols:
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            ser_clean = ser.dropna()
            n = int(ser_clean.count()) if not ser_clean.empty else 0
            mean = float(ser_clean.mean()) if n>0 else np.nan
            median = float(ser_clean.median()) if n>0 else np.nan
            std = float(ser_clean.std()) if n>0 else np.nan
            mn = float(ser_clean.min()) if n>0 else np.nan
            mx = float(ser_clean.max()) if n>0 else np.nan
            q1 = float(ser_clean.quantile(0.25)) if n>0 else np.nan
            q3 = float(ser_clean.quantile(0.75)) if n>0 else np.nan
            iqr = q3 - q1 if n>0 else np.nan
            lower = q1 - 1.5 * iqr if n>0 else np.nan
            upper = q3 + 1.5 * iqr if n>0 else np.nan
            outliers = ser_clean[(ser_clean < lower) | (ser_clean > upper)] if n>0 else ser_clean.iloc[0:0]
            n_out = int(outliers.count())
            pct_out = n_out / n if n>0 else 0
            skewness = float(ser_clean.skew()) if n>2 else 0.0

            # description
            mean_f = f"{mean:.2f}" if not np.isnan(mean) else "nan"
            median_f = f"{median:.2f}" if not np.isnan(median) else "nan"
            std_f = f"{std:.2f}" if not np.isnan(std) else "nan"
            mn_f = f"{mn:.2f}" if not np.isnan(mn) else "nan"
            mx_f = f"{mx:.2f}" if not np.isnan(mx) else "nan"
            skew_f = f"{skewness:.2f}"
            desc = (
                f'Count={n}; mean={mean_f}; median={median_f}; std={std_f}; '
                f'min={mn_f}; max={mx_f}; outliers={n_out} ({pct_out:.1%}). '
                f'Skewness={skew_f}.'
            )

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8.5,11), facecolor='white', gridspec_kw={'height_ratios':[3,1]})
            ax_hist = axes[0]
            sns.histplot(ser_clean, kde=True, ax=ax_hist)
            ax_hist.set_title(f'Histogram: {col}')
            ax_hist.set_xlabel(col)
            ax_box = axes[1]
            sns.boxplot(x=ser_clean, orient='h', ax=ax_box)
            ax_box.set_title(f'Boxplot: {col}')
            ax_box.set_xlabel('')
            fig.patch.set_facecolor('white')
            fig.text(0.01, 0.02, textwrap.fill(desc, 200), ha='left', va='bottom', fontsize=9)
            plt.tight_layout(rect=[0,0.06,1,0.98])

            # save images
            fname_hist = out_plots_dir / f"{sanitize(col)}_hist.png"
            fname_box = out_plots_dir / f"{sanitize(col)}_box.png"
            fig.savefig(fname_hist, bbox_inches='tight', facecolor=fig.get_facecolor())
            # small box-only image
            fig_box_small, axb = plt.subplots(figsize=(8,3), facecolor='white')
            sns.boxplot(x=ser_clean, orient='h', ax=axb)
            axb.set_title(f'Boxplot: {col}')
            plt.tight_layout()
            fig_box_small.savefig(fname_box, bbox_inches='tight', facecolor=fig_box_small.get_facecolor())
            plt.close(fig_box_small)

            pp.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)

        else:
            ser_cat = ser.fillna('<<MISSING>>').astype(str)
            vc = ser_cat.value_counts().head(50)
            fig = plt.figure(figsize=(8.5,11), facecolor='white')
            ax = fig.add_subplot(111)
            sns.barplot(y=vc.index.astype(str), x=vc.values, palette='viridis', ax=ax)
            ax.set_title(f'Value counts: {col}')
            ax.set_xlabel('Count')
            ax.set_ylabel(col)
            total = int(ser_cat.shape[0])
            lines = []
            for k, v in vc.items():
                pct = v / total if total>0 else 0
                lines.append(f'{k}: {v} ({pct:.1%})')
            top_pct = vc.iloc[0] / total if total>0 and len(vc)>0 else 0
            if top_pct > 0.8:
                importance = 'Single category dominates — low variability.'
            elif top_pct > 0.4:
                importance = 'Top category is sizeable — consider grouping smaller categories.'
            else:
                importance = 'Healthy spread across categories.'
            desc_full = 'Top categories: ' + ' | '.join(lines) + f'. Note: {importance}'
            fig.patch.set_facecolor('white')
            fig.text(0.01, 0.02, '\n'.join(textwrap.wrap(desc_full, 300)), ha='left', va='bottom', fontsize=9)
            plt.tight_layout(rect=[0,0.06,1,0.98])
            fname_bar = out_plots_dir / f"{sanitize(col)}_bar.png"
            fig.savefig(fname_bar, bbox_inches='tight', facecolor=fig.get_facecolor())
            pp.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)

    pp.close()


def main():
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('InputData/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized.csv')
    if not inp.exists():
        print('Input file not found:', inp)
        sys.exit(2)

    df = pd.read_csv(inp, engine='python', encoding='utf-8')
    print('Read', inp, 'shape', df.shape)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    print('Numeric cols detected:', len(numeric))

    df_noout, keep_mask = remove_outliers(df, numeric)
    out_csv = inp.parent / (inp.stem + '-NoOutliers' + inp.suffix)
    df_noout.to_csv(out_csv, index=False, encoding='utf-8')
    print('Saved outlier-removed CSV to', out_csv, 'shape', df_noout.shape)

    # new output folder
    out_plots_dir = inp.parent / 'outputs_nooutliers' / 'plots'
    pdf_out = inp.parent / 'scales_plots_report_nooutliers.pdf'
    print('Rendering plots to', out_plots_dir, 'and PDF', pdf_out)
    plot_all_columns(df_noout, out_plots_dir, pdf_out)
    # write small index
    with open(inp.parent / 'outputs_nooutliers' / 'plots' / 'index.txt', 'w', encoding='utf-8') as fh:
        for p in sorted((inp.parent / 'outputs_nooutliers' / 'plots').iterdir()):
            fh.write(p.name + '\n')

    print('Done.')


if __name__ == '__main__':
    main()
