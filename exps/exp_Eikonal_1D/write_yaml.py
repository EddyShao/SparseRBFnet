import copy
import itertools
from pathlib import Path

import yaml  # pip install pyyaml

# --------- user choices here ----------
BASE_CFG_PATH = Path("exps/exp_Eikonal_1D/base.yaml")
OUT_DIR = Path("exps/exp_Eikonal_1D/")

# epsilons you want to try
EPSILONS = [0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 1/1280, 1/2560]
# epsilon_name = [20, 40, 80, 160, 320, 640]

# (Nobs_int, Nobs_bnd) pairs you want to try
OBS_PAIRS = [(29, 2), (59, 2), (119, 2), (249, 2), (499, 2), (999, 2), (1999, 2), (3999, 2), (7999, 2), (15999, 2), (31999, 2)]
obs_name = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
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
        fname = f"eikonal_eps{int(1 / eps):04d}_N_{int(Nint+1):05d}.yaml"
        out_path = OUT_DIR / fname

        with out_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()