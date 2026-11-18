"""Aggregate runs/*/results.json into a markdown table (mean over seeds)."""
import glob
import json
from collections import defaultdict
from statistics import mean

groups = defaultdict(list)
for p in glob.glob("runs/*/results.json"):
    r = json.load(open(p))
    a = r["args"]
    groups[(a["w1"], a["w2"])].append(r)

cols = ["Chat", "Chat Hard", "Safety", "Reasoning", "Avg"]
head = "| w1 | w2 | #seeds | " + " | ".join(f"F:{c}" for c in cols) + " | L:Avg | HS2 val MSE |"
lines = [head, "|" + "---|" * (head.count("|") - 1)]

for (w1, w2), rs in sorted(groups.items(), key=lambda kv: (-kv[0][0], -kv[0][1])):
    f = [mean(r["rewardbench"]["F"].get(c, float("nan")) for r in rs) for c in cols]
    l_avg = mean(r["rewardbench"]["L"]["Avg"] for r in rs)
    mse = mean(mean(r["helpsteer2_val_mse"].values()) for r in rs)
    lines.append(f"| {w1} | {w2} | {len(rs)} | " + " | ".join(f"{x:.1f}" for x in f)
                 + f" | {l_avg:.1f} | {mse:.2f} |")

out = "\n".join(lines)
print(out)
open("results_table.md", "w").write(out + "\n")