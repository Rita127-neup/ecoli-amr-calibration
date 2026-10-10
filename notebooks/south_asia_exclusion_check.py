"""
South Asia sensitivity check: refit the UK elastic-net model, then evaluate South Asia
with and without the BioProject PRJNA925003 (35 genomes, all resistant).
Needs only the public repo file data/processed/qc_cohort_final_features.csv.

Colab: new notebook -> paste this whole file into one cell -> Run.
Saves south_asia_exclusion_check.txt next to the notebook (Files panel, left side).
"""
import sys, io, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings('ignore')

SRC = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.csv') else
       'https://raw.githubusercontent.com/Rita127-neup/ecoli-amr-calibration/main/'
       'data/processed/qc_cohort_final_features.csv')
out = io.StringIO()
def say(s=''):
    print(s); out.write(s + '\n')

d = pd.read_csv(SRC, dtype={'Genome ID': str})
d['y'] = (d['Resistant Phenotype'] == 'Resistant').astype(int)
d['Isolation Country'] = d['Isolation Country'].replace({'England': 'United Kingdom'})
F = ['gyrA_fq_mut', 'parC_fq_mut', 'qnrA', 'qnrB', 'qnrS', 'oqxA', 'oqxB', 'aac6_Ib_cr']
uk = d[(d['Isolation Country'] == 'United Kingdom') & d['MLST'].notna()]
sa = d[d['Isolation Country'].isin(['India', 'Pakistan', 'Bangladesh'])]
nor = d[d['Isolation Country'] == 'Norway']
say(f'UK training n={len(uk)} (expect 2574) | South Asia n={len(sa)} (expect 238) | Norway n={len(nor)} (expect 3153)')

clf = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', C=1.0,
                         max_iter=5000, random_state=0).fit(uk[F].values, uk['y'].values)

def slope_int(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return round(float(m.coef_[0][0]), 3), round(float(m.intercept_[0]), 3)

def report(name, df):
    p = clf.predict_proba(df[F].values)[:, 1]
    s, i = slope_int(df['y'], p)
    say(f"{name:34s} n={len(df):4d} resistant={df['y'].sum():4d} prev={df['y'].mean():.3f} "
        f"meanP={p.mean():.3f} AUROC={roc_auc_score(df['y'], p):.3f} slope={s} intercept={i} "
        f"Brier={brier_score_loss(df['y'], p):.4f}")

say('\nSouth Asia by BioProject (size, resistant, share resistant):')
say(sa.groupby('BioProject Accession', dropna=False)['y'].agg(['size', 'sum', 'mean']).round(3)
      .sort_values('size', ascending=False).to_string())
say('\nPaper values to match: Norway slope 0.726 / intercept -0.488; South Asia slope 0.786 / intercept 0.927\n')
report('Norway (full)', nor)
report('South Asia (full)', sa)
report('South Asia excl. PRJNA925003', sa[sa['BioProject Accession'] != 'PRJNA925003'])
say('\nlibraries: ' + ', '.join(f'{m.__name__} {m.__version__}' for m in (pd, np, __import__('sklearn'))))
open('south_asia_exclusion_check.txt', 'w').write(out.getvalue())
