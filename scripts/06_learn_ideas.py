"""Learn the idea dictionary from population mistakes, with a split-half stability check.

Only *positional* mistakes are used: the position is still contested (best-move win prob
inside ideas.contested) and the two settled line ends have about the same material
(|gap| < ideas.tactical_material). Decided-position and tactical mistakes are counted
but not clustered (separate tracks). The sweep also fits column-shuffled null data:
atoms that are as stable on shuffled data as on real data are single-feature artifacts.

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
from coach.ideas import (IdeaModel, column_shuffle, delta_scale, fit_nmf, make_inputs, match_atoms,
                         n80, split_by_game, track)
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
    d_all = D["dB"] - D["dP"]
    tr = track(df.wp_best.to_numpy(), d_all, names, icfg)
    counts = pd.Series(tr[mistake]).replace("", "unusable").value_counts()
    print(f"{mistake.sum():,} population mistakes: "
          + ", ".join(f"{k} {v:,} ({v / mistake.sum():.0%})" for k, v in counts.items()))
    use = mistake & (tr == "positional")
    X, keep, norms = make_inputs(d_all[use], scale, icfg["min_norm"])
    sub = df[use].reset_index(drop=True)[keep].reset_index(drop=True)
    print(f"learning positional ideas from {len(X):,} mistakes "
          f"(wp_best in {icfg['contested']}, |material gap| < {icfg['tactical_material']})")
    if len(X) < 10 * max(icfg["sweep"] + [icfg["n_atoms"]]):
        print("WARNING: very few samples per atom; results will be noisy.")

    aw, ah = icfg["alpha_W"], icfg.get("alpha_H", 0.0)
    half = split_by_game(sub.game_id.to_numpy(), icfg["seed"])
    Xn = column_shuffle(X, icfg["seed"])

    def stability(Z, k):
        full = fit_nmf(Z, k, icfg["seed"], aw, ah)
        a = fit_nmf(Z[half], k, icfg["seed"], aw, ah)
        b = fit_nmf(Z[~half], k, icfg["seed"], aw, ah)
        return full, np.minimum(match_atoms(full.components_, a.components_),
                                match_atoms(full.components_, b.components_))

    rows, final = [], None
    max_k = min(int(half.sum()), int((~half).sum()), X.shape[1])
    if icfg["n_atoms"] > max_k:
        sys.exit(f"ideas.n_atoms = {icfg['n_atoms']} but each half has only {max_k} usable rows")
    for k in sorted(set(icfg["sweep"]) | {icfg["n_atoms"]}):
        if k > max_k:
            print(f"  K={k:3d}: skipped (needs more than {max_k} rows per half)")
            continue
        full, stab = stability(X, k)
        _, stab_null = stability(Xn, k)
        st = stab >= icfg["stable_cos"]
        n_stable = int(st.sum())
        n_null = int((stab_null >= icfg["stable_cos"]).sum())
        err = full.reconstruction_err_ / np.linalg.norm(X)
        simp = n80(full.components_)
        med_n80 = float(np.median(simp[st])) if st.any() else float("nan")
        rows.append({"k": k, "stable_atoms": n_stable, "null_stable_atoms": n_null,
                     "rel_recon_error": round(float(err), 4),
                     "median_stability": round(float(np.median(stab)), 3),
                     "median_n80_stable": med_n80})
        print(f"  K={k:3d}: {n_stable:3d} stable (null data: {n_null:3d}), "
              f"recon error {err:.3f}, median n80 of stable atoms {med_n80:.0f}")
        if k == icfg["n_atoms"]:
            final = (full, stab)

    out = data_dir(cfg, "ideas")
    pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)
    nmf, stab = final
    model = IdeaModel(nmf=nmf, scale=scale, feature_names=names, min_norm=icfg["min_norm"],
                      stability=stab, stable_cos=icfg["stable_cos"],
                      meta={"n_train": len(X), "rating_center": cfg["population"]["rating_center"],
                            "contested": list(icfg["contested"]),
                            "tactical_material": icfg["tactical_material"],
                            "alpha_H": ah, "track_counts": counts.to_dict()})
    model.save(out / "model.pkl")
    W = nmf.transform(X)
    np.savez_compressed(out / "pop_codes.npz", pos_id=sub.pos_id.to_numpy().astype(str), W=W,
                        norm=norms[keep])
    print(f"saved K={model.k} model ({int((stab >= icfg['stable_cos']).sum())} stable atoms) -> {out}")


if __name__ == "__main__":
    main()
