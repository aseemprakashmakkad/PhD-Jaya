#!/usr/bin/env python3
"""
Normalize `Monthly household income` into numeric midpoint and category.
Saves a new CSV with two added columns:
 - Monthly household income_midpoint (numeric, INR)
 - Monthly household income_cat (one of: 'Below 10k','10k-30k','30k-50k','50k-100k','Above 100k','Unknown')

Assumptions:
 - Ranges like '10,000 - 30,000 INR per month' are parsed and midpoint taken (20000)
 - 'Below 10,000' -> midpoint 5000
 - 'Above 100,000' -> midpoint 125000 (proxy)
 - Commas and extra text are ignored where possible

Usage: python3 normalize_income.py /path/to/cleaned.csv
"""
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/pranav/PhD-Jaya/InputData/20251214-ScalesData-Combined_ver0.7-Cleaned.csv')
OUT_PATH = IN_PATH.parent / (IN_PATH.stem + '-IncomeNormalized' + IN_PATH.suffix)

if not IN_PATH.exists():
    print('Input file not found:', IN_PATH)
    sys.exit(2)

# read csv (already cleaned header)
df = pd.read_csv(IN_PATH, engine='python', encoding='utf-8')

col = 'Monthly household income'
if col not in df.columns:
    print(f"Column '{col}' not found in {IN_PATH}")
    sys.exit(3)

# helper to parse a string and return midpoint and category
range_rx = re.compile(r"(?P<low>\d+[\d,]*)\s*[-–—]\s*(?P<high>\d+[\d,]*)")
num_rx = re.compile(r"(?P<num>\d+[\d,]*)")

def parse_income(s):
    if pd.isna(s):
        return (np.nan, 'Unknown')
    s0 = str(s).strip()
    # normalize whitespace
    s0 = re.sub(r"\s+", ' ', s0)
    # handle common words
    s_low = s0.lower()
    # 'below' or 'less than'
    if 'below' in s_low or 'less than' in s_low or 'under' in s_low:
        m = num_rx.search(s)
        if m:
            val = int(m.group('num').replace(',', ''))
            midpoint = val / 2.0
            cat = 'Below 10k' if val <= 10000 else 'Below ' + f'{val:,}'
            return (midpoint, cat)
    # 'above' or 'more than'
    if 'above' in s_low or 'more than' in s_low or 'over' in s_low:
        m = num_rx.search(s)
        if m:
            val = int(m.group('num').replace(',', ''))
            # proxy midpoint for open-ended top bin
            midpoint = val * 1.25
            cat = 'Above 100k' if val >= 100000 else 'Above ' + f'{val:,}'
            return (midpoint, cat)
    # range like 10,000 - 30,000
    m = range_rx.search(s)
    if m:
        low = int(m.group('low').replace(',', ''))
        high = int(m.group('high').replace(',', ''))
        midpoint = (low + high) / 2.0
        # assign category based on the detected range
        if high <= 10000:
            cat = 'Below 10k'
        elif low >= 100000:
            cat = 'Above 100k'
        elif low >= 50000:
            cat = '50k-100k'
        elif low >= 30000:
            cat = '30k-50k'
        elif low >= 10000:
            cat = '10k-30k'
        else:
            cat = f'{low:,}-{high:,}'
        return (midpoint, cat)
    # single number
    m = num_rx.search(s)
    if m:
        val = int(m.group('num').replace(',', ''))
        # treat single numbers as category based on magnitude
        if val < 10000:
            cat = 'Below 10k'
        elif val < 30000:
            cat = '10k-30k'
        elif val < 50000:
            cat = '30k-50k'
        elif val < 100000:
            cat = '50k-100k'
        else:
            cat = 'Above 100k'
        return (val, cat)
    # otherwise unknown
    return (np.nan, 'Unknown')

# Apply parser
midpoints = []
cats = []
for v in df[col].values:
    m, c = parse_income(v)
    midpoints.append(m)
    cats.append(c)

df[col + '_midpoint_INR'] = midpoints
# normalize category naming to consistent set
cat_map = {
    'Below 10k': 'Below 10k',
    '10k-30k': '10k-30k',
    '30k-50k': '30k-50k',
    '50k-100k': '50k-100k',
    'Above 100k': 'Above 100k'
}
# Some entries used different capitalization; map where possible
cats_norm = []
for c in cats:
    # try to map ranges like '10,000 - 30,000 INR per month' which we returned as '10k-30k'
    if isinstance(c, str) and c in cat_map:
        cats_norm.append(cat_map[c])
    else:
        # try to map a label by checking numbers
        if isinstance(c, str) and c.lower().startswith('below'):
            cats_norm.append('Below 10k')
        elif isinstance(c, str) and c.lower().startswith('above'):
            cats_norm.append('Above 100k')
        elif c == 'Unknown':
            cats_norm.append('Unknown')
        else:
            cats_norm.append(c)

df[col + '_cat'] = cats_norm

# Save
df.to_csv(OUT_PATH, index=False, encoding='utf-8')
print('Saved normalized CSV to:', OUT_PATH)
# Print counts
print('\nCategory counts:')
print(df[col + '_cat'].value_counts(dropna=False).to_string())

# Update: also output a small summary file
summary = {
    'input': str(IN_PATH),
    'output': str(OUT_PATH),
    'n_rows': int(len(df)),
}

from json import dump
with open(OUT_PATH.parent / (OUT_PATH.stem + '-income-summary.json'), 'w', encoding='utf-8') as fh:
    dump(summary, fh, indent=2)

print('\nDone.')
