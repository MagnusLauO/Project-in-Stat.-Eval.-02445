#!/usr/bin/env python3
# ============================================================
# anova_check.py – 3-vejs ANOVA + antagelsestest
# ------------------------------------------------------------
#   gender × age_band × occupation  → middle_salary
# ------------------------------------------------------------
#   Afhængigheder: pandas, numpy, scipy, statsmodels, matplotlib
#   pip install pandas numpy scipy statsmodels matplotlib
# ============================================================

import argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, levene
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import statsmodels.stats.multicomp as mc


# ---------- Data & model ------------------------------------
def load_and_prepare(csv_path: pathlib.Path, width: int = 5, start: int = 25):
    """Indlæs CSV og læg alder i 'width'-års-bånd (fx 25-29, 30-34 …)."""
    cols = ["gender", "age", "occupation", "middle_salary"]
    df = pd.read_csv(csv_path, usecols=cols)

    bins   = list(range(start, int(df["age"].max()) + width, width))
    labels = [f"{b}-{b+width-1}" for b in bins[:-1]]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    for col in ["gender", "age_band", "occupation"]:
        df[col] = df[col].astype("category")

    return df.dropna(subset=["age_band"])


def fit_anova(df):
    model = smf.ols("middle_salary ~ C(gender)*C(age_band)*C(occupation)", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return model, table


# ---------- Antagelsestest ----------------------------------
def normality(resid):
    w, p = shapiro(resid)
    print(f"Shapiro-Wilk: W = {w:.3f}, p = {p:.4g}")


def homoscedasticity(df):
    groups = df.groupby(["gender", "age_band", "occupation"])["middle_salary"].apply(list)

    Fm, pm = levene(*groups, center="mean")       # klassisk Levene
    Fbf, pbf = levene(*groups, center="median")   # Brown-Forsythe

    print(f"Levene (mean-center) : F = {Fm:.3f}, p = {pm:.4g}")
    print(f"Brown-Forsythe       : F = {Fbf:.3f}, p = {pbf:.4g}")


def heteroskedasticity(model):
    lm, lmp, f, fp = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan        : LM = {lm:.2f}, p = {lmp:.4g}")


def independence(resid):
    dw = durbin_watson(resid)
    print(f"Durbin-Watson        : DW = {dw:.3f}")


# ---------- Plots (frivillige) ------------------------------
def residual_plots(resid):
    z = (resid - resid.mean()) / resid.std(ddof=1)
    sm.qqplot(z, line="45")
    plt.title("Q–Q-plot af standardiserede residualer")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.hist(z, bins="auto", edgecolor="k")
    plt.xlabel("Standardiseret residual")
    plt.title("Histogram over residualer")
    plt.tight_layout(); plt.show()


def cooks_plot(model, cutoff=None):
    infl = model.get_influence()
    c, _ = infl.cooks_distance
    if cutoff is None:
        cutoff = 4 / len(c)

    plt.figure(figsize=(8, 3))
    plt.stem(c, markerfmt=",", basefmt=" ")
    plt.axhline(cutoff, ls="--")
    plt.title(f"Cook’s distance (cutoff ≈ {cutoff:.3f})")
    plt.xlabel("Observation")
    plt.tight_layout(); plt.show()

    print(f"{(c > cutoff).sum()} observation(er) ligger over cutoff.")


# ---------- Post-hoc ----------------------------------------
def tukey_posthoc(df):
    df = df.copy()
    df["grp"] = df["age_band"].astype(str) + "_" + df["gender"].astype(str)
    res = mc.MultiComparison(df["middle_salary"], df["grp"]).tukeyhsd()
    print("\n--- Tukey HSD (alder×køn) ---")
    print(res.summary())


# ---------- Main --------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="3-vejs ANOVA med diagnostik")
    ap.add_argument("csv", type=str, help="CSV-fil med løndata")
    ap.add_argument("--bin", type=int, default=5,
                    help="Alders-bånd bredde (år), default 5")
    ap.add_argument("--no-plots", action="store_true",
                    help="Spring alle plots over")
    ap.add_argument("--posthoc", action="store_true",
                    help="Kør Tukey HSD post-hoc")
    args = ap.parse_args()

    df = load_and_prepare(args.csv, args.bin)
    model, tbl = fit_anova(df)

    print("\n=== 3-vejs ANOVA (Type II) ===")
    print(tbl.loc[["C(gender)", "C(age_band)",
                   "C(occupation)", "C(gender):C(age_band)"]])

    print("\n--- Antagelsestest ---")
    normality(model.resid)
    homoscedasticity(df)
    heteroskedasticity(model)
    independence(model.resid)

    if not args.no_plots:
        residual_plots(model.resid)
        cooks_plot(model)

    if args.posthoc:
        tukey_posthoc(df)


if __name__ == "__main__":
    main()