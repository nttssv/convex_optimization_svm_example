#!/usr/bin/env python3
"""Convergence analysis for the gradient-descent squared-hinge SVM.

Measures how many iterations gradient descent needs to reach the optimum f* at
several tolerances and learning rates, and plots the optimality gap f(k)-f* on a
log scale to show linear convergence. Writes:
    output/figures/convergence.png
    output/figures/convergence_iterations.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEAT=["PAC","PRA","Potassium","Tumor size","18-OHF","18-oxoF","Systolic BP","Diastolic BP","DDD","Age"]
LOGF=["PAC","PRA","18-OHF","18-oxoF"]
LAM=1e-2

def load():
    df=pd.read_csv("data/cgh_pa_dataset.csv")
    y=(df["Model 1 Target"]=="UPA").astype(int).to_numpy()
    X=df[FEAT].apply(pd.to_numeric,errors="coerce")
    for c in LOGF: X[c]=np.log(X[c]+1e-6)
    X=X.fillna(X.median()); X=((X-X.mean())/X.std(ddof=0)).to_numpy()
    return X,y

def cweights(y):
    n=len(y); cw=np.ones(n)
    for c in (0,1):
        m=(y==c); cw[m]=n/(2*m.sum())
    return cw

def run(X,y,t,cw,lr,iters):
    n=len(y); w=np.zeros(X.shape[1]); b=0.0; hist=np.empty(iters)
    for k in range(iters):
        margin=1-t*(X@w+b); a=margin>0
        hist[k]=0.5*LAM*w@w+np.sum(cw*np.maximum(0,margin)**2)/n
        gw=LAM*w; gb=0.0
        if a.any():
            wv=cw[a]*margin[a]; gw-=(2/n)*(X[a].T@(wv*t[a])); gb-=(2/n)*np.sum(wv*t[a])
        w-=lr*gw; b-=lr*gb
    return hist

def main():
    X,y=load(); t=2*y-1.0; cw=cweights(y)
    U=t[:,None]*np.column_stack([X,np.ones(len(y))])
    H=np.zeros((11,11)); H[:10,:10]=LAM*np.eye(10); H+=(2/len(y))*(U.T@(cw[:,None]*U))
    L=float(np.linalg.eigvalsh(H)[-1]); safe=1/L
    fstar=run(X,y,t,cw,0.403354,60000).min()

    mults=[0.05,0.10,0.25,0.50,1.00,1.90]
    rows=[]; curves={}
    for mm in mults:
        h=run(X,y,t,cw,mm*safe,2000); curves[mm]=h
        row={"lr_mult_of_1overL":mm,"learning_rate":mm*safe}
        for tol in (1e-2,1e-4,1e-6):
            idx=np.where(h-fstar<=tol)[0]
            row[f"iters_tol_{tol:g}"]=int(idx[0]) if len(idx) else None
        rows.append(row)
    out=Path("output/figures"); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"convergence_iterations.csv",index=False)

    plt.figure(figsize=(7.0,4.4))
    for mm in mults:
        gap=np.maximum(curves[mm]-fstar,1e-12)
        plt.semilogy(gap[:400],linewidth=1.7,label=f"{mm:g}/L")
    plt.axhline(1e-4,color="black",ls="--",lw=1,label=r"$10^{-4}$ tolerance")
    plt.xlabel("Gradient descent iteration $k$")
    plt.ylabel(r"Optimality gap $f(\theta_k)-f^\star$ (log scale)")
    plt.title("Convergence of gradient descent (squared-hinge SVM)")
    plt.grid(alpha=0.25,which="both"); plt.legend(ncol=2,fontsize=8)
    plt.tight_layout(); plt.savefig(out/"convergence.png",dpi=220); plt.close()
    print(f"f*={fstar:.6f}  L={L:.4f}  1/L={safe:.4f}")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":
    main()
