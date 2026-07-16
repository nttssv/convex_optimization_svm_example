#!/usr/bin/env python3
"""Nested-free stratified 5-fold CV AUROC/metrics for the gradient-descent SVM.

Uses the same squared-hinge linear SVM optimizer as the diagnostics script, with
fold-wise log-transform + standardization to avoid leakage. Reports the model's
OWN performance on the (synthetic) CGH cohort.
"""
from __future__ import annotations
import numpy as np, pandas as pd

FEATURES = ["PAC","PRA","Potassium","Tumor size","18-OHF","18-oxoF","Systolic BP","Diastolic BP","DDD","Age"]
LOGF = ["PAC","PRA","18-OHF","18-oxoF"]

def fit_gd(X,y,lam=1e-2,lr=0.2,iters=3000):
    t=2*y-1.0; w=np.zeros(X.shape[1]); b=0.0; n=len(y)
    cw=np.ones(n)
    for c in (0,1):
        m=(y==c); 
        if m.sum(): cw[m]=n/(2*m.sum())
    for _ in range(iters):
        margin=1-t*(X@w+b); a=margin>0
        gw=lam*w; gb=0.0
        if a.any():
            wv=cw[a]*margin[a]
            gw-= (2/n)*(X[a].T@(wv*t[a])); gb-=(2/n)*np.sum(wv*t[a])
        w-=lr*gw; b-=lr*gb
    return w,b

def auroc(y,s):
    # rank-based AUROC
    order=np.argsort(s); r=np.empty(len(s)); r[order]=np.arange(1,len(s)+1)
    # handle ties by average rank
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    csum=np.cumsum(cnt); start=csum-cnt
    avg=(start+csum+1)/2.0
    r=avg[inv]
    n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

df=pd.read_csv("data/cgh_pa_dataset.csv")
y=(df["Model 1 Target"]=="UPA").astype(int).to_numpy()
Xraw=df[FEATURES].apply(pd.to_numeric,errors="coerce")

rng=np.random.default_rng(0)
idx0=np.where(y==0)[0].copy(); idx1=np.where(y==1)[0].copy()
rng.shuffle(idx0); rng.shuffle(idx1)
folds=[[] for _ in range(5)]
for i,ix in enumerate(idx0): folds[i%5].append(ix)
for i,ix in enumerate(idx1): folds[i%5].append(ix)
folds=[np.array(f) for f in folds]

scores=np.zeros(len(y)); 
for k in range(5):
    te=folds[k]; tr=np.concatenate([folds[j] for j in range(5) if j!=k])
    Xtr=Xraw.iloc[tr].copy(); Xte=Xraw.iloc[te].copy()
    for c in LOGF:
        off=1e-6; Xtr[c]=np.log(Xtr[c]+off); Xte[c]=np.log(Xte[c]+off)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xte=Xte.fillna(med)
    mu=Xtr.mean(); sd=Xtr.std(ddof=0).replace(0,1.0)
    Xtr=((Xtr-mu)/sd).to_numpy(); Xte=((Xte-mu)/sd).to_numpy()
    w,b=fit_gd(Xtr,y[tr])
    scores[te]=Xte@w+b

overall=auroc(y,scores)
pred=(scores>=0).astype(int)
tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
sens=tp/(tp+fn); spec=tn/(tn+fp); prec=tp/(tp+fp) if tp+fp else 0
f1=2*prec*sens/(prec+sens) if prec+sens else 0
print(f"CV AUROC     : {overall:.3f}")
print(f"Sensitivity  : {sens:.3f}")
print(f"Specificity  : {spec:.3f}")
print(f"Precision    : {prec:.3f}")
print(f"F1 score     : {f1:.3f}")
with open("output/figures/svm_cv_metrics.txt","w") as fh:
    fh.write(f"AUROC {overall:.3f}\nSensitivity {sens:.3f}\nSpecificity {spec:.3f}\nPrecision {prec:.3f}\nF1 {f1:.3f}\n")
