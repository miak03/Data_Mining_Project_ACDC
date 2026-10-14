import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/mnt/user-data/uploads/acdc_dataset.csv", sep=";").drop_duplicates()
df["lifetime_value"] = pd.to_numeric(df["lifetime_value"].astype(str).str.replace("€", "").str.replace(",", "").str.strip(), errors="coerce")

num_cols = ["age", "n_colleagues_on_account", "plan_days_per_week", "days_since_last_visit",
            "total_visits", "avg_weekly_visits", "entitled_visits_last_period",
            "actual_visits_last_period", "meeting_room_hours", "event_space_hours",
            "guest_passes_issued", "n_renewals", "lifetime_value"]

stats = pd.DataFrame({
    "missing": df[num_cols].isna().sum(),
    "min": df[num_cols].min(),
    "median": df[num_cols].median(),
    "mean": df[num_cols].mean(),
    "max": df[num_cols].max(),
    "skew": df[num_cols].skew(),
    "kurtosis": df[num_cols].kurt(),
    "pct_zero": (df[num_cols] == 0).mean() * 100,
    "n_negative": (df[num_cols] < 0).sum(),
}).round(2)
print(stats.to_string())
stats.to_csv("numeric_stats.csv")

fig, axes = plt.subplots(4, 4, figsize=(18, 14))
for ax, col in zip(axes.flat, num_cols):
    s = df[col].dropna()
    lo, hi = s.quantile([0.005, 0.995])
    s_clip = s[(s >= lo) & (s <= hi)]
    ax.hist(s_clip, bins=40, color="#4C72B0", edgecolor="white")
    ax.set_title(f"{col}\nskew={s.skew():.2f}  out of view={len(s) - len(s_clip)}", fontsize=10)
for ax in axes.flat[len(num_cols):]:
    ax.axis("off")
fig.suptitle("Numeric distributions (raw, deduplicated, central 99% shown)", fontsize=14)
fig.tight_layout()
fig.savefig("numeric_distributions.png", dpi=110)