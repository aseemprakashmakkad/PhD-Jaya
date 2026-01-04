#!/usr/bin/env python3
"""
Relational analysis between a set of independent and dependent variables.

Reads a cleaned NoOutliers CSV (default path under InputData) and:
- computes correlations (Pearson/Spearman) for numeric vs numeric pairs
- computes ANOVA/Kruskal or group comparisons for categorical vs numeric
- computes chi-square for categorical vs categorical
- creates plots for each pair and saves them under an outputs folder
- writes a CSV summary of statistical test results and a PDF report

Usage:
  python3 scripts/relational_analysis.py /path/to/NoOutliers.csv

Outputs (defaults):
  - InputData/outputs_relation/plots/
  - InputData/relation_summary.csv
  - InputData/relation_report.pdf

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
import warnings
import math

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import statsmodels.api as sm
    from statsmodels.stats import multitest
except Exception:
    sm = None
    multitest = None

sns.set(style='whitegrid')


INDEP = [
    'Age (IN YEARS)',
    'Marital Status',
    'Highest level of education completed',
    'Current Employment Status',
    'Number of children',
    'Type of family structure',
    'Monthly household income',
    'Access to healthcare services',
    'Overall health status',
    'Are you covered by health insurance?',
    'Participation in community or social activities',
    'Functional Impairment',
    'Withdrawl',
    'Occupational & Relationship Consequences ',
    'Compulsive Behaviour ',
    'Obsession with Internet',
    'Internet as a Source of Recreation',
    'Enhanced Socialization ',
    'Perceived Control of Internet Use ',
    'Monthly household income_midpoint_INR',
    'Monthly household income_cat'
]

DEPEND = [
    'Emotional Support',
    'Informational Support',
    'Instrumental Support',
    'Perceived Social Support TOTAL SCORE',
    'Emotional Support - T Score',
    'Informational Support - T Score',
    'Instrumental Support - T Score',
    'Perceived Social Support TOTAL T SCORE',
    'Emotional Support - Rank',
    'Informational Support - Rank',
    'Instrumental Support - Rank',
    'TOTAL SCORE - Rank',
    'Real Self Concept Total',
    'Ideal Self Concept Total',
    'Social Self Concept Total',
    'Real Self Concept Total.1',
    'Ideal Self Concept Total.1',
    'Social Self Concept Total.1',
    'Real Self Concept Total.2',
    'Ideal Self Concept Total.2',
    'Social Self Concept Total.2',
    'Real Self Concept Total.3',
    'Ideal Self Concept Total.3',
    'Social Self Concept Total.3',
    'Real Self Concept Total.4',
    'Ideal Self Concept Total.4',
    'Social Self Concept Total.4',
    'Real Self Concept Total.5',
    'Ideal Self Concept Total.5',
    'Social Self Concept Total.5',
    'Real Self Concept Total Category ',
    'Ideal Self Concept Total Category',
    'Social Self Concept Total Category ',
    'Unnamed: 80',
    'Unnamed: 81'
]


def sanitize(s):
    return ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in str(s)).strip().replace(' ', '_')


def is_categorical(series):
    # treat object and category dtypes as categorical, also small-unique numeric columns
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return True
    if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) < 20:
        return True
    return False


def analyze_pair(df, xcol, ycol, outdir, pdf):
    x = df[xcol]
    y = df[ycol]
    res = {'independent': xcol, 'dependent': ycol, 'test': None, 'statistic': None, 'pvalue': np.nan, 'notes': ''}

    # decide types
    x_cat = is_categorical(x)
    y_cat = is_categorical(y)

    # Create a plot per pair and add to pdf
    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    ax = fig.add_subplot(111)

    try:
        if (not x_cat) and (not y_cat):
            # numeric-numeric: Pearson and Spearman, scatter with reg line
            # dropna pairwise
            df_xy = df[[xcol, ycol]].dropna()
            if df_xy.shape[0] < 5:
                res['notes'] = 'Too few observations'
            else:
                if stats is not None:
                    try:
                        pear_r, pear_p = stats.pearsonr(df_xy[xcol], df_xy[ycol])
                        spea_res = stats.spearmanr(df_xy[xcol], df_xy[ycol])
                        spea_r, spea_p = spea_res.correlation, spea_res.pvalue
                        res['test'] = 'pearson/spearman'
                        res['statistic'] = pear_r
                        res['pvalue'] = pear_p
                        res['notes'] = f'spearman_r={spea_r:.3f}, spearman_p={spea_p:.3g}'
                    except Exception as e:
                        res['notes'] = 'scipy correlation failed: ' + str(e)
                        res['statistic'] = df_xy[xcol].corr(df_xy[ycol])
                        res['pvalue'] = np.nan
                else:
                    res['notes'] = 'scipy not available; using pandas corr'
                    res['statistic'] = df_xy[xcol].corr(df_xy[ycol])
                    res['pvalue'] = np.nan

                # main scatter with overall regression
                sns.regplot(x=xcol, y=ycol, data=df_xy, ax=ax, scatter_kws={'s':10}, line_kws={'color':'red'})
                ax.set_title(f'{ycol} vs {xcol} (overall)')

                # if marital status exists, add a small stratified subplot image to show group-wise trends
                if 'Marital Status' in df.columns:
                    try:
                        fig2 = plt.figure(figsize=(8.5, 3), facecolor='white')
                        ax2 = fig2.add_subplot(111)
                        sns.scatterplot(x=xcol, y=ycol, hue='Marital Status', data=df_xy, ax=ax2, s=20)
                        # per-group regression lines
                        for name, grp in df_xy.groupby('Marital Status'):
                            if grp.shape[0] > 2:
                                try:
                                    sns.regplot(x=xcol, y=ycol, data=grp, ax=ax2, scatter=False, label=str(name))
                                except Exception:
                                    pass
                        ax2.set_title('By Marital Status')
                        # save it temporarily
                        fname2 = outdir / f"{sanitize(xcol)}__vs__{sanitize(ycol)}__by_Marital_Status.png"
                        fig2.savefig(fname2, bbox_inches='tight', facecolor=fig2.get_facecolor())
                        plt.close(fig2)
                    except Exception:
                        pass

        elif x_cat and (not y_cat):
            # categorical x, numeric y: boxplot and ANOVA/Kruskal
            df_xy = df[[xcol, ycol]].dropna()
            if df_xy.shape[0] < 5:
                res['notes'] = 'Too few observations'
            else:
                order = list(df_xy.groupby(xcol)[ycol].median().sort_values(ascending=False).index)
                sns.boxplot(x=xcol, y=ycol, data=df_xy, order=order, ax=ax)
                ax.set_title(f'{ycol} by {xcol}')
                # statistical test
                groups = [g[ycol].values for n, g in df_xy.groupby(xcol)]
                try:
                    if stats is not None:
                        anova = stats.f_oneway(*groups)
                        res['test'] = 'ANOVA'
                        res['statistic'] = anova.statistic
                        res['pvalue'] = anova.pvalue
                    else:
                        res['notes'] = 'scipy not available'
                except Exception as e:
                    # fallback to Kruskal
                    try:
                        kr = stats.kruskal(*groups)
                        res['test'] = 'Kruskal'
                        res['statistic'] = kr.statistic
                        res['pvalue'] = kr.pvalue
                    except Exception as e2:
                        res['notes'] = 'ANOVA/Kruskal failed: ' + str(e2)

        elif (not x_cat) and y_cat:
            # numeric x, categorical y: swap roles and do similar
            df_xy = df[[xcol, ycol]].dropna()
            if df_xy.shape[0] < 5:
                res['notes'] = 'Too few observations'
            else:
                order = list(df_xy.groupby(ycol)[xcol].median().sort_values(ascending=False).index)
                sns.boxplot(x=ycol, y=xcol, data=df_xy, order=order, ax=ax)
                ax.set_title(f'{xcol} by {ycol}')
                groups = [g[xcol].values for n, g in df_xy.groupby(ycol)]
                try:
                    anova = stats.f_oneway(*groups)
                    res['test'] = 'ANOVA'
                    res['statistic'] = anova.statistic
                    res['pvalue'] = anova.pvalue
                except Exception as e:
                    try:
                        kr = stats.kruskal(*groups)
                        res['test'] = 'Kruskal'
                        res['statistic'] = kr.statistic
                        res['pvalue'] = kr.pvalue
                    except Exception as e2:
                        res['notes'] = 'ANOVA/Kruskal failed: ' + str(e2)

        else:
            # categorical-categorical: contingency table and chi-square
            df_xy = df[[xcol, ycol]].dropna()
            if df_xy.shape[0] < 5:
                res['notes'] = 'Too few observations'
            else:
                ct = pd.crosstab(df_xy[xcol], df_xy[ycol])
                sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Cross-tab: {xcol} x {ycol}')
                try:
                    chi2, p, dof, ex = stats.chi2_contingency(ct)
                    res['test'] = 'chi2'
                    res['statistic'] = chi2
                    res['pvalue'] = p
                except Exception as e:
                    res['notes'] = 'chi2 test failed: ' + str(e)

    except Exception as e:
        res['notes'] = 'Plot/test error: ' + str(e)

    # save plot file
    fname = outdir / f"{sanitize(xcol)}__vs__{sanitize(ycol)}.png"
    fig.savefig(fname, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    # append to pdf
    try:
        # create a simple figure to add to pdf from saved image to avoid backend issues
        img = plt.imread(str(fname))
        fig2 = plt.figure(figsize=(8.5, 11), facecolor='white')
        ax2 = fig2.add_subplot(111)
        ax2.axis('off')
        ax2.imshow(img)
        pp = pdf
        pp.savefig(fig2, bbox_inches='tight', facecolor=fig2.get_facecolor())
        plt.close(fig2)
    except Exception:
        pass

    # If we created a marital-status stratified image (fname2), add it to the PDF as well
    try:
        if 'fname2' in locals() and fname2.exists():
            img2 = plt.imread(str(fname2))
            fig3 = plt.figure(figsize=(8.5, 3), facecolor='white')
            ax3 = fig3.add_subplot(111)
            ax3.axis('off')
            ax3.imshow(img2)
            pdf.savefig(fig3, bbox_inches='tight', facecolor=fig3.get_facecolor())
            plt.close(fig3)
    except Exception:
        pass

    return res


def main():
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('InputData/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized-NoOutliers.csv')
    if not inp.exists():
        print('Input file not found:', inp)
        sys.exit(2)

    df = pd.read_csv(inp, engine='python', encoding='utf-8')
    print('Loaded', inp, 'shape', df.shape)

    # pick only columns that exist
    indep = [c for c in INDEP if c in df.columns]
    depend = [c for c in DEPEND if c in df.columns]
    print('Independent columns:', len(indep), 'Dependent columns:', len(depend))

    out_plots = inp.parent / 'outputs_relation' / 'plots'
    out_plots.mkdir(parents=True, exist_ok=True)
    pdf_out = inp.parent / 'relation_report.pdf'
    summary_rows = []

    pp = PdfPages(str(pdf_out))

    # Loop through pairs and analyze
    for x in indep:
        for y in depend:
            print('Analyzing:', x, '->', y)
            r = analyze_pair(df, x, y, out_plots, pp)
            summary_rows.append(r)

    pp.close()

    # write summary CSV
    out_summary = inp.parent / 'relation_summary.csv'
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print('Wrote summary to', out_summary)
    print('Plots to', out_plots)
    print('PDF to', pdf_out)


if __name__ == '__main__':
    main()
