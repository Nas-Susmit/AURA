"""Compare regenerated CSV outputs with the manuscript's archived outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def main():
    report={}
    for f in sorted((ROOT/'reference_results').glob('*.csv')):
        a=pd.read_csv(f); b=pd.read_csv(ROOT/'revision_results'/f.name)
        assert a.shape==b.shape and list(a.columns)==list(b.columns),f.name
        errors=[]
        for k in a.columns:
            if pd.api.types.is_numeric_dtype(a[k]):
                np.testing.assert_allclose(a[k],b[k],atol=1e-9,rtol=1e-8,equal_nan=True,err_msg=f'{f.name}: {k}')
                dif=np.abs(a[k]-b[k]); errors.append(float(dif.max()) if dif.notna().any() else 0.)
            else: pd.testing.assert_series_equal(a[k],b[k])
        report[f.name]={'rows':len(a),'max_absolute_difference':max(errors,default=0.)}
    d=pd.read_csv(ROOT/'revision_results/metrics.csv')
    assert len(d)==1400 and not d.duplicated(['seed','scenario','model','ranking']).any()
    assert d.seed.nunique()==10 and d.scenario.nunique()==7
    assert len(pd.read_csv(ROOT/'revision_results/sensitivity.csv'))==630
    for _,g in d.groupby(['seed','scenario','model']):
        for k in ['auroc','auprc','brier','ece','accuracy']:
            np.testing.assert_allclose(g[k],g[k].iloc[0],atol=1e-12)
    print(json.dumps(report,indent=2));print('PASS: all saved manuscript CSVs reproduced within tolerance.')
if __name__=='__main__': main()
