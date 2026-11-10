"""Learn the idea dictionary from population mistakes, with a split-half stability check.

Inputs: data/population/deltas.npz + engine results.
Outputs:
  data/ideas/model.pkl        the final dictionary (ideas.n_atoms atoms) + scaling
  data/ideas/sweep.csv        how many atoms survive the stability check for each K
  data/ideas/pop_codes.npz    atom loadings for every population mistake used

    python scripts/06_learn_ideas.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.ideas import IdeaModel, delta_scale, fit_nmf, make_inputs, match_atoms, split_by_game
from coach.results import load_with_deltas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    icfg = cfg["ideas"]

    df, D, names = load_with_deltas(cfg, "population")
    scale = delta_scale(D["dB"], D["dP"], D["dA"])
    mistake = (df.label == "mistake").to_numpy() & ~df.played_is_best.to_numpy()
    d = (D["dB"] - D["dP"])[mistake]
    X, keep, norms = make_inputs(d, scale, icfg["min_norm"])
    sub = df[mistake].reset_index(drop=True)[keep].reset_index(drop=True)
    print(f"{mistake.sum():,} population mistakes; {len(X):,} with usable, non-trivial deltas")
    if len(X) < 10 * max(icfg["sweep"] + [icfg["n_atoms"]]):
        print("WARNING: very few samples per atom; results will be noisy.")

    half = split_by_game(sub.game_id.to_numpy(), icfg["seed"])
    rows, final = [], None
    for k in sorted(set(icfg["sweep"]) | {icfg["n_atoms"]}):
        full = fit_nmf(X, k, icfg["seed"], icfg["alpha_W"])
        a = fit_nmf(X[half], k, icfg["seed"], icfg["alpha_W"])
        b = fit_nmf(X[~half], k, icfg["seed"], icfg["alpha_W"])
        stab = np.minimum(match_atoms(full.components_, a.components_),
                          match_atoms(full.components_, b.components_))
        n_stable = int((stab >= icfg["stable_cos"]).sum())
        err = full.reconstruction_err_ / np.linalg.norm(X)
        rows.append({"k": k, "stable_atoms": n_stable, "rel_recon_error": round(float(err), 4),
                     "median_stability": round(float(np.median(stab)), 3)})
        print(f"  K={k:3d}: {n_stable:3d} stable atoms (cos >= {icfg['stable_cos']}), "
              f"relative reconstruction error {err:.3f}")
        if k == icfg["n_atoms"]:
            final = (full, stab)

    out = data_dir(cfg, "ideas")
    pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)
    nmf, stab = final
    model = IdeaModel(nmf=nmf, scale=scale, feature_names=names, min_norm=icfg["min_norm"],
                      stability=stab, stable_cos=icfg["stable_cos"],
                      meta={"n_train": len(X), "rating_center": cfg["population"]["rating_center"]})
    model.save(out / "model.pkl")
    W = nmf.transform(X)
    np.savez_compressed(out / "pop_codes.npz", pos_id=sub.pos_id.to_numpy().astype(str), W=W,
                        norm=norms[keep])
    print(f"saved K={model.k} model ({int((stab >= icfg['stable_cos']).sum())} stable atoms) -> {out}")


if __name__ == "__main__":
    main()
