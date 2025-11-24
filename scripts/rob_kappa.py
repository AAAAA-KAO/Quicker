import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm.auto import trange

np.random.seed(42)

# Load dataframe
df = pd.read_excel("RA_rob_label_with_llm.xlsx")
df["uid"] = df["file"].astype(str) + "___" + df["outcome_uid"].astype(str)

binary_domains = ["allocation", "blinding", "data", "selective_reporting"]
overall_labels = [0, -1, -2]

pairs = [("label", "pred"), ("label", "assessor"), ("pred", "assessor")]

pair_names = {
    ("label", "pred"): "Guideline vs Quicker",
    ("label", "assessor"): "Guideline vs ExpertRep",
    ("pred", "assessor"): "Quicker vs ExpertRep",
}


# -----------------------------
# Bootstrap CI function
# -----------------------------
def bootstrap_kappa(a, b, weights=None, n_boot=1000):
    n = len(a)
    idx = np.arange(n)
    boot_vals = []

    for _ in range(n_boot):
        sample_idx = np.random.choice(idx, size=n, replace=True)
        try:
            k = cohen_kappa_score(a[sample_idx], b[sample_idx], weights=weights)
        except:
            k = np.nan
        boot_vals.append(k)

    boot_vals = np.array(boot_vals)
    est = cohen_kappa_score(a, b, weights=weights)
    lo = np.nanpercentile(boot_vals, 2.5)
    hi = np.nanpercentile(boot_vals, 97.5)
    return est, lo, hi


# -----------------------------
# Compute results
# -----------------------------
records = []

# --- Binary domains ---
for domain in binary_domains:
    for src1, src2 in pairs:
        f1 = f"{src1}_{domain}"
        f2 = f"{src2}_{domain}"

        a = df[f1].astype(int).values
        b = df[f2].astype(int).values

        est, lo, hi = bootstrap_kappa(a, b, weights=None, n_boot=2000)

        cm = confusion_matrix(a, b, labels=[0, 1]).tolist()

        records.append(
            {
                "domain": domain,
                "type": "binary",
                "pair": pair_names[(src1, src2)],
                "kappa": est,
                "ci_lower": lo,
                "ci_upper": hi,
                "confusion_matrix": cm,
            }
        )

# --- Overall ordinal domain ---
for src1, src2 in pairs:
    f1 = f"{src1}_rob_score"
    f2 = f"{src2}_rob_score"

    a = df[f1].astype(int).values
    b = df[f2].astype(int).values

    est, lo, hi = bootstrap_kappa(a, b, weights="quadratic", n_boot=2000)
    cm = confusion_matrix(a, b, labels=overall_labels).tolist()

    records.append(
        {
            "domain": "overall",
            "type": "ordinal",
            "pair": pair_names[(src1, src2)],
            "kappa": est,
            "ci_lower": lo,
            "ci_upper": hi,
            "confusion_matrix": cm,
        }
    )

results_df = pd.DataFrame(records)
results_df.to_csv("rob_kappa_results.csv", index=False)
