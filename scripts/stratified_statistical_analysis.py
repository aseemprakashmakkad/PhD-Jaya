"""
stratified_statistical_analysis.py

Performs statistical analysis (Spearman, ANOVA, Chi-square) between internet-related exposures, demographic covariates, and dependent variables, stratified by marital status (married vs. unmarried). Summarizes key findings and generates a well-formatted MS Word document.

Usage:
    python3 scripts/stratified_statistical_analysis.py relation_summary_with_fdr.csv InputData/20251214-ScalesData-Combined_ver0.7-Cleaned-IncomeNormalized.csv

Outputs:
    outputs/stratified_statistical_analysis.docx

Dependencies:
    pandas, numpy, python-docx
"""
import sys
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def summarize_spearman(df):
    rows = []
    for _, row in df[df['test'].str.contains('spearman', na=False)].iterrows():
        r = row['statistic']
        p = row['pvalue']
        indep = row['independent']
        dep = row['dependent']
        if pd.notnull(r) and pd.notnull(p):
            if abs(r) >= 0.5:
                strength = 'strong'
            elif abs(r) >= 0.3:
                strength = 'moderate'
            elif abs(r) >= 0.1:
                strength = 'weak'
            else:
                strength = 'very weak'
            direction = 'positive' if r > 0 else 'negative'
            sig = 'statistically significant' if p < 0.05 else 'not significant'
            comment = f"{indep} vs. {dep}: {strength} {direction} association (r={r:.2f}, p={p:.3g}, {sig})"
            rows.append(comment)
    return rows

def summarize_anova(df):
    rows = []
    for _, row in df[df['test'].str.contains('anova', na=False)].iterrows():
        F = row['statistic']
        p = row['pvalue']
        indep = row['independent']
        dep = row['dependent']
        if pd.notnull(F) and pd.notnull(p):
            strength = 'strong' if F >= 5 else 'moderate' if F >= 2 else 'weak'
            sig = 'statistically significant' if p < 0.05 else 'not significant'
            comment = f"{indep} vs. {dep}: {strength} group differences (F={F:.2f}, p={p:.3g}, {sig})"
            rows.append(comment)
    return rows

def summarize_chi2(df):
    rows = []
    for _, row in df[df['test'].str.contains('chi2', na=False)].iterrows():
        chi2 = row['statistic']
        p = row['pvalue']
        indep = row['independent']
        dep = row['dependent']
        if pd.notnull(chi2) and pd.notnull(p):
            strength = 'strong' if chi2 >= 10 else 'moderate' if chi2 >= 5 else 'weak'
            sig = 'statistically significant' if p < 0.05 else 'not significant'
            comment = f"{indep} vs. {dep}: {strength} non-random association (χ²={chi2:.2f}, p={p:.3g}, {sig})"
            rows.append(comment)
    return rows

def main():
    # Analyze both married and single group summary files
    summary_files = [
        ("Married", Path("outputs_internet/married/relation_summary_with_fdr.csv")),
        ("Single", Path("outputs_internet/single/relation_summary_with_fdr.csv")),
    ]
    doc = Document()
    doc.add_heading("Stratified Statistical Analysis by Marital Status", 0)
    for group, summary_csv in summary_files:
        doc.add_heading(f"{group} Females", 1)
        if not summary_csv.exists():
            doc.add_paragraph(f"No summary file found for {group} females.")
            continue
        df_summary = pd.read_csv(summary_csv)
        # Summarize results for this group
        results = []
        for _, row in df_summary.iterrows():
            indep = row['independent']
            dep = row['dependent']
            test = row['test']
            stat = row['statistic']
            p = row['pvalue']
            if pd.isnull(stat) or pd.isnull(p):
                continue
            if test == 'spearman':
                if abs(stat) >= 0.5:
                    strength = 'strong'
                elif abs(stat) >= 0.3:
                    strength = 'moderate'
                elif abs(stat) >= 0.1:
                    strength = 'weak'
                else:
                    strength = 'very weak'
                direction = 'positive' if stat > 0 else 'negative'
                sig = 'statistically significant' if p < 0.05 else 'not significant'
                comment = f"{indep} vs. {dep}: {strength} {direction} association (r={stat:.2f}, p={p:.3g}, {sig})"
                results.append(comment)
            elif test == 'anova':
                strength = 'strong' if stat >= 5 else 'moderate' if stat >= 2 else 'weak'
                sig = 'statistically significant' if p < 0.05 else 'not significant'
                comment = f"{indep} vs. {dep}: {strength} group differences (F={stat:.2f}, p={p:.3g}, {sig})"
                results.append(comment)
            elif test == 'chi2':
                strength = 'strong' if stat >= 10 else 'moderate' if stat >= 5 else 'weak'
                sig = 'statistically significant' if p < 0.05 else 'not significant'
                comment = f"{indep} vs. {dep}: {strength} non-random association (χ²={stat:.2f}, p={p:.3g}, {sig})"
                results.append(comment)
        if results:
            for c in results:
                doc.add_paragraph(c, style='List Bullet')
        else:
            doc.add_paragraph("No significant results for this group.")
    doc.save('outputs/stratified_statistical_analysis.docx')
    print('Saved stratified analysis to outputs/stratified_statistical_analysis.docx')

if __name__ == '__main__':
    main()
