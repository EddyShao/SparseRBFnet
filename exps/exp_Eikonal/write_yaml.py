import copy
import itertools
from pathlib import Path

import yaml  # pip install pyyaml

# --------- user choices here ----------
BASE_CFG_PATH = Path("exps/exp_Eikonal/base.yaml")
OUT_DIR = Path("exps/exp_Eikonal/")

# epsilons you want to try
EPSILONS = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125]
epsilon_name = [10, 20, 40, 80, 160, 320, 640, 1280]

# (Nobs_int, Nobs_bnd) pairs you want to try
OBS_PAIRS = [(784, 116), (3364, 236), (7744, 316), (13924, 476)]
obs_name = [30, 60, 90, 120]
# --------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with BASE_CFG_PATH.open("r") as f:
        base_cfg = yaml.safe_load(f)

    for eps, (Nint, Nbnd) in itertools.product(EPSILONS, OBS_PAIRS):
        cfg = copy.deepcopy(base_cfg)

        # modify PDE block
        cfg["pde"]["epsilon"] = float(eps)
        cfg["pde"]["Nobs_int"] = int(Nint)
        cfg["pde"]["Nobs_bnd"] = int(Nbnd)

        # nice filename encoding the choices
        fname = f"eikonal_eps{int(1 / eps):04d}_N_{int(Nint**.5 + 2):03d}.yaml"
        out_path = OUT_DIR / fname

        with out_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()