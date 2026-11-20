import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Melyik futtatás melyik csoportba tartozik: egy csoport = egy vonal a grafikonon
RUNS = {
    "PPO": ["walker_norm", "ppo_s1", "ppo_s2"],
    "SAC": ["sac_s0", "sac_s1", "sac_s2"],
}
COLORS = {"PPO": "tab:blue", "SAC": "tab:orange"}

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="eval/mean_reward", help="melyik TensorBoard-görbét rajzolja")
parser.add_argument("--x", default="steps", choices=["steps", "time"], help="vízszintes tengely: lépések vagy eltelt percek")
parser.add_argument("--out", default="plots/ppo_vs_sac.png")
args = parser.parse_args()


def load_curve(run_name, tag):
    # A TensorBoard-mappa neve a futtatás neve és egy sorszám, például logs/ppo_s1_1
    folders = sorted(f for f in glob.glob(f"logs/{run_name}_*") if re.fullmatch(rf"{run_name}_\d+", os.path.basename(f)))
    if not folders:
        raise SystemExit(f"Nincs ilyen napló: logs/{run_name}_*")
    events = EventAccumulator(folders[-1])      # ha többször futott, a legutóbbi
    events.Reload()
    points = events.Scalars(tag)
    if args.x == "time":
        # a napló minden pontnál az időt is tárolja: percek a futtatás indulása óta
        steps = np.array([(p.wall_time - events.FirstEventTimestamp()) / 60 for p in points])
    else:
        steps = np.array([p.step for p in points])
    values = np.array([p.value for p in points])
    return steps, values


unit = "perc" if args.x == "time" else "lépés"
plt.figure(figsize=(8, 4.5))
for label, run_names in RUNS.items():
    curves = [load_curve(name, args.tag) for name in run_names]
    # A futtatások nem mindig pontosan ugyanott mértek, ezért közös rácsra hozzuk őket
    first_common_step = max(steps[0] for steps, _ in curves)
    last_common_step = min(steps[-1] for steps, _ in curves)
    grid = np.linspace(first_common_step, last_common_step, 200)
    values = np.array([np.interp(grid, steps, vals) for steps, vals in curves])

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    color = COLORS[label]
    for row in values:                           # halványan az egyes seedek
        plt.plot(grid, row, color=color, alpha=0.25, linewidth=1)
    plt.plot(grid, mean, color=color, linewidth=2, label=f"{label} (n={len(curves)})")
    plt.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)
    print(f"{label}: végső átlag {mean[-1]:.0f} +/- {std[-1]:.0f} ({len(curves)} seed, {last_common_step:,.0f} {unit}nél)")

plt.xlabel("Training time (minutes)" if args.x == "time" else "Environment steps")
plt.ylabel("Mean evaluation return")
plt.title("Walker2d-v5: PPO vs SAC (mean ± std over seeds)")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Grafikon elmentve: {args.out}")