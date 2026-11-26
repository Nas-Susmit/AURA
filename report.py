"""Generate concise reviewer-facing summaries from the latest run."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parent
    source=root/'revision_results'; out=root/'reports'; out.mkdir(exist_ok=True)
    d=pd.read_csv(source/'metrics.csv')
    d.groupby(['model','ranking'])[['auroc','auprc','brier','ece','aurc','risk80']].mean().to_csv(out/'macro_summary.csv')
    s=pd.read_csv(source/'sensitivity.csv').groupby(['variant','seed']).delta.mean().unstack()
    mu=s.mean(axis=1); h=t.ppf(.975,s.shape[1]-1)*s.std(axis=1,ddof=1)/np.sqrt(s.shape[1])
    table=pd.DataFrame({'delta_aurc':mu,'low':mu-h,'high':mu+h});table.to_csv(out/'sensitivity_intervals.csv')
    fig,ax=plt.subplots(figsize=(8,4.5));ax.errorbar(mu,range(len(mu)),xerr=h,fmt='o',capsize=3);ax.set_yticks(range(len(mu)),mu.index);ax.axvline(0,ls='--',color='black');ax.set(xlabel='AURC change: full − no D',title='Synthetic benchmark sensitivity');fig.tight_layout();fig.savefig(out/'sensitivity.png',dpi=180);plt.close(fig)
    print(table.round(7).to_string()); print('Report saved to',out)
if __name__=='__main__': main()
