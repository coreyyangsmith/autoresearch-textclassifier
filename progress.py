import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

RESULTS_FILE = "results.tsv"
OUTPUT_FILE = "progress.png"
METRIC = "val_f1_macro"
METRIC_LABEL = "Validation F1 Macro (higher is better)"

rows = []
with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

# Assign experiment numbers (0-indexed, excluding crashes from the numbered sequence)
# but keep crashes visible as markers. We number all rows sequentially.
experiments = []
for i, row in enumerate(rows):
    status = row["status"].strip().lower()
    try:
        metric = float(row[METRIC])
    except ValueError:
        metric = None

    # Crashes and zero-value rows are excluded from performance tracking
    valid = status in ("keep", "discard") and metric is not None and metric > 0.0
    experiments.append({
        "idx": i,
        "commit": row["commit"].strip(),
        "metric": metric if valid else None,
        "status": status,
        "description": row["description"].strip(),
        "valid": valid,
    })

# Build running best from kept experiments only
running_best = []
best_so_far = None
best_x = []
best_y = []

for exp in experiments:
    if exp["valid"] and exp["status"] == "keep":
        if best_so_far is None or exp["metric"] > best_so_far:
            best_so_far = exp["metric"]
        best_x.append(exp["idx"])
        best_y.append(best_so_far)

# Extend running best line to last kept point
# Build a step-function: for each kept point, the best extends horizontally
step_x = []
step_y = []
cur_best = None
for exp in experiments:
    if exp["valid"] and exp["status"] == "keep":
        if cur_best is None or exp["metric"] > cur_best:
            cur_best = exp["metric"]
        step_x.append(exp["idx"])
        step_y.append(cur_best)

n_total = len(experiments)
n_kept = sum(1 for e in experiments if e["status"] == "keep" and e["valid"])
n_improvements = sum(
    1 for i, e in enumerate(experiments)
    if e["status"] == "keep" and e["valid"] and (
        i == 0 or e["metric"] > max(
            (x["metric"] for x in experiments[:i] if x["status"] == "keep" and x["valid"]),
            default=0,
        )
    )
)

# Separate into groups
kept_x = [e["idx"] for e in experiments if e["status"] == "keep" and e["valid"]]
kept_y = [e["metric"] for e in experiments if e["status"] == "keep" and e["valid"]]
discard_x = [e["idx"] for e in experiments if e["status"] == "discard" and e["valid"]]
discard_y = [e["metric"] for e in experiments if e["status"] == "discard" and e["valid"]]
crash_x = [e["idx"] for e in experiments if e["status"] == "crash"]

GREEN = "#2ca02c"
GREY = "#aaaaaa"
CRASH_COLOR = "#d62728"

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#f8f8f8")
ax.set_facecolor("#f8f8f8")

# Discarded experiments
ax.scatter(discard_x, discard_y, color=GREY, alpha=0.5, s=20, zorder=2, label="Discarded")

# Crashes as small x markers along the bottom of the plot
if crash_x:
    y_min_for_crash = ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else min(kept_y + discard_y) - 0.005
    crash_y_vals = [min(kept_y + discard_y) - 0.012] * len(crash_x)
    ax.scatter(crash_x, crash_y_vals, color=CRASH_COLOR, alpha=0.6, s=20,
               marker="x", zorder=2, label="Crash")

# Kept experiments
ax.scatter(kept_x, kept_y, color=GREEN, s=50, zorder=4, label="Kept")

# Running best step line
if step_x:
    # Draw a step function: for each consecutive pair of kept points, draw horizontal then vertical
    line_x = []
    line_y = []
    for i in range(len(step_x)):
        line_x.append(step_x[i])
        line_y.append(step_y[i])
        if i + 1 < len(step_x):
            line_x.append(step_x[i + 1])
            line_y.append(step_y[i])
    ax.plot(line_x, line_y, color=GREEN, linewidth=1.5, zorder=3, label="Running best")

# Annotate kept experiments that are new bests
cur_best = None
for exp in experiments:
    if exp["status"] == "keep" and exp["valid"]:
        is_new_best = cur_best is None or exp["metric"] > cur_best
        if is_new_best:
            cur_best = exp["metric"]
        label_text = exp["description"]
        if len(label_text) > 40:
            label_text = label_text[:38] + "…"
        ax.annotate(
            label_text,
            xy=(exp["idx"], exp["metric"]),
            xytext=(5, 12),
            textcoords="offset points",
            fontsize=5.5,
            color=GREEN,
            rotation=35,
            va="bottom",
            ha="left",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )

ax.set_xlabel("Experiment #", fontsize=11)
ax.set_ylabel(METRIC_LABEL, fontsize=11)
ax.set_title(
    f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept",
    fontsize=13,
    fontweight="bold",
)
ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_xlim(-1, n_total + 1)

# Tight y-axis around the valid data
all_valid_y = kept_y + discard_y
if all_valid_y:
    y_lo = min(all_valid_y) - 0.005
    y_hi = max(all_valid_y) + 0.005
    ax.set_ylim(y_lo, y_hi)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
print(f"Saved {OUTPUT_FILE}")
