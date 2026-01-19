import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys
sys.path.append("./")
import argparse
import os
import pickle

import subprocess
import threading

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from src.config.base_config import load_config_from_args
from src.config.exp_config import (
    SolverGlobalConfig,
    SolverPhaseConfig,
    HomotopyConfig,
)
from src.config.output_config import (
    SolverOutput,
    PhaseResult,
    LevelResult,
    ExperimentResult,
)
from src.solver.SOLVER_REGISTRY import SOLVER_REGISTRY
from src.utils import Objective, compute_errors, compute_rhs, compute_rhs_aux
from pde.PDE_REGISTRY import build_pde_from_cfg  


def _try_nvml_handle():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml, h
    except Exception:
        return None, None

def _gpu_mem_used_mb_nvml(pynvml, h):
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return info.used / (1024**2)

def _gpu_mem_used_mb_nvidia_smi():
    # returns memory.used in MiB for GPU 0
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader", "-i", "0"],
        text=True
    ).strip()
    return float(out.splitlines()[0])

class PeakMemSampler:
    def __init__(self, interval_sec=0.002):
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self.peak_mb = 0.0
        self._thread = None

        self._pynvml, self._h = _try_nvml_handle()
        self._use_nvml = (self._pynvml is not None)

    def _read_mb(self):
        if self._use_nvml:
            return _gpu_mem_used_mb_nvml(self._pynvml, self._h)
        else:
            return _gpu_mem_used_mb_nvidia_smi()

    def _run(self):
        while not self._stop.is_set():
            try:
                mb = self._read_mb()
                if mb > self.peak_mb:
                    self.peak_mb = mb
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self):
        self.peak_mb = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_mb

def _block(x):
    return jax.block_until_ready(x)

def benchmark_sparse_rbf_bilap(
    kernel,
    d: int,
    B: int = 1024,
    K_list=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
    reps: int = 50,
    warmup: int = 2,
    seed: int = 0,
    save_path: str = "ref_solvers/sparserbf_bilap_eval_runonly_mem_runonly.npz",
):
    import os

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    Xhat = jax.random.uniform(sub, shape=(B, d), minval=-1.0, maxval=1.0)

    sampler = PeakMemSampler(interval_sec=0.002)

    t_bilap_ms = []
    peak_mem_mb = []

    # JIT once; recompilation happens automatically when X/S/c shapes change
    @jax.jit
    def bilap_eval(X, S, c, Xhat):
        return kernel.BiLap_kappa_X_c_Xhat(X, S, c, Xhat)  # (B,)

    for K in K_list:
        # "Padding" for this run is exactly K (i.e., Kmax == K)
        key, k1, k2, k3 = jax.random.split(key, 4)
        X = jax.random.uniform(k1, shape=(K, d), minval=-1.0, maxval=1.0)
        S = jax.random.normal(k2, shape=(K, 1))  # adjust if anisotropic
        c = jax.random.normal(k3, shape=(K,))

        # ---- Compile (excluded): first call for this shape triggers compilation
        _block(bilap_eval(X, S, c, Xhat))

        # ---- Warmup (excluded): stabilize caches / autotuning
        for _ in range(warmup):
            _block(bilap_eval(X, S, c, Xhat))

        # ---- Measure EXECUTION ONLY (time + peak mem), excluding compile/warmup
        sampler.start()
        t0 = time.perf_counter()

        for _ in range(reps):
            y = bilap_eval(X, S, c, Xhat)
            _block(y)

        t1 = time.perf_counter()
        peak = sampler.stop()

        avg_ms = 1000.0 * (t1 - t0) / reps
        t_bilap_ms.append(avg_ms)
        peak_mem_mb.append(peak)

        print(f"K={K:6d} | run-only={avg_ms:8.3f} ms | peak_mem(run-only)={peak:8.1f} MiB")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        K_list=np.array(K_list),
        t_bilap_ms=np.array(t_bilap_ms),
        peak_mem_mb=np.array(peak_mem_mb),
        B=np.array(B),
        d=np.array(d),
        reps=np.array(reps),
        warmup=np.array(warmup),
        fp64=np.array(jax.config.read("jax_enable_x64")),
        device=np.array(str(jax.devices()[0])),
    )
    print(f"Saved -> {save_path}")
# ---------- Argument parsing ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Generic PDE solver driver.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="Random seed, override config (if given).")
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override save_dir from YAML (optional).",
    )
    return parser.parse_args()




# ---------- Solver schedule (phases + homotopy) ----------

def build_alg_opts(global_cfg: SolverGlobalConfig,
                   phase_cfg: SolverPhaseConfig,
                   cfg) -> dict:
    """Build the flat dict of options expected by your solvers."""
    opts = {
        "alpha": global_cfg.alpha,
        "T": global_cfg.T,
        "gamma": global_cfg.gamma,
        "Ntrial": global_cfg.Ntrial,
        "blocksize": global_cfg.blocksize,
        "print_every": global_cfg.print_every,
        "plot_every": global_cfg.plot_every,
        "max_step": phase_cfg.max_step,
        "TOL": phase_cfg.TOL,
        "plot_final": phase_cfg.plot_final,
        # PDE-related options that some solvers expect:
        "anisotropic": bool(cfg.kernel.anisotropic),
        "scale": float(cfg.pde.scale),
        "sampling": str(cfg.pde.sampling),
        "seed": int(cfg.pde.seed),
    }
    return opts





# ---------- Main ----------

def main():
    args = parse_args()
    cfg = load_config_from_args(args)
    if args.seed is not None:
        # assuming Config stores underlying dict in .data or similar
        cfg.pde.data["seed"] = args.seed

    jax.config.update("jax_enable_x64", getattr(cfg.solver, "fp64", True))
    print("jax_enable_x64 =", jax.config.read("jax_enable_x64"))
    print(f'DATA TYPE SANITY CHECK: {jnp.ones((16,)).dtype}')

    print("Loaded config:")
    print(cfg)


    # Build PDE from config
    p = build_pde_from_cfg(cfg)

    kernel = p.kernel

    res = benchmark_sparse_rbf_bilap(
        kernel=kernel,
        d=p.d,
        B=1024,
        K_list=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        reps=20,
        warmup=2,
        # out_prefix=args.save_dir if args.save_dir is not None else "exps/bilap_vis",
    )

main()  
