#!/usr/bin/env python3
"""
Invert T–μB EOS tables to (e, nB) grids and export P, T, μB, speed of sound squared (cs2)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
from scipy.interpolate import RectBivariateSpline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# ------------------------------------------------------------------#
# Global constants
# ------------------------------------------------------------------#
# Convergence criteria for root-finding
ACCURACY: float = 1.0e-6
# Maximum number of iterations for bisection
MAXITER: int = 100
# Planck's constant times speed of light [GeV·fm]
HBAR_C: float = 0.19733

# EOS grid definitions for (e, nB) tiling
ED_BOUNDS: List[float] = [0.0, 0.0036, 0.015, 0.045, 0.455, 20.355, 650.0]
NE_LIST:   List[int]   = [61,   60,    61,    122,    200,     400]
# Maximum nB range and points for each table
NB_BOUNDS: List[float] = [0.0025, 0.015, 0.045, 0.25, 3.5, 12.0]
NNB_LIST:  List[int]   = [501,   301,   181,   251, 351,  251]

# Rich console for progress output
console = Console()


# ------------------------------------------------------------------#
# Helpers: interpolation wrappers and root finders
# ------------------------------------------------------------------#

def build_interpolants(
    ed_raw: np.ndarray,
    nB_raw: np.ndarray,
    P_raw: np.ndarray,
    T_axis: np.ndarray,
    muB_axis: np.ndarray,
) -> Tuple[RectBivariateSpline, RectBivariateSpline, RectBivariateSpline]:
    """
    Construct 2D cubic spline interpolants:
      - f_e(T, μB) → energy density e
      - f_nB(T, μB) → baryon density nB
      - f_P(T, μB) → pressure P
    """
    f_e  = RectBivariateSpline(T_axis, muB_axis, ed_raw, kx=3, ky=3)
    f_nB = RectBivariateSpline(T_axis, muB_axis, nB_raw, kx=3, ky=3)
    f_P  = RectBivariateSpline(T_axis, muB_axis, P_raw,  kx=3, ky=3)
    return f_e, f_nB, f_P


def solve_T(
    ed_target: float,
    muB_val: float,
    f_e: RectBivariateSpline,
    T_bounds: Tuple[float, float] = (0.03, 0.80), # in GeV
) -> float:
    """
    Find temperature T such that f_e(T, μB_val) = ed_target
    using 1D bisection.
    """
    lo, hi = T_bounds
    e_lo = float(f_e.ev(lo, muB_val))
    e_hi = float(f_e.ev(hi, muB_val))
    if ed_target <= e_lo:
        return lo
    if ed_target >= e_hi:
        return hi
    for _ in range(MAXITER):
        mid = 0.5 * (lo + hi)
        e_mid = float(f_e.ev(mid, muB_val))
        if abs(e_mid - ed_target) < ACCURACY * max(1.0, abs(ed_target)):
            return mid
        if ed_target < e_mid:
            hi = mid
        else:
            lo = mid
    return mid


def solve_T_muB(
    ed_target: float,
    nB_target: float,
    f_e: RectBivariateSpline,
    f_nB: RectBivariateSpline,
    muB_bounds: Tuple[float, float] = (0.0, 0.45), # in GeV
    T_bounds: Tuple[float, float] = (0.03, 0.80), # in GeV
) -> Tuple[float, float]:
    """
    Solve coupled e(T, μB)=ed_target and nB(T, μB)=nB_target
    by nested bisection: outer on μB, inner on T.
    """
    lo, hi = muB_bounds
    T_lo = solve_T(ed_target, lo, f_e, T_bounds)
    T_hi = solve_T(ed_target, hi, f_e, T_bounds)
    nB_lo = float(f_nB.ev(T_lo, lo))
    nB_hi = float(f_nB.ev(T_hi, hi))
    if nB_target <= nB_lo:
        return T_lo, lo
    if nB_target >= nB_hi:
        return T_hi, hi
    for _ in range(MAXITER):
        mid = 0.5 * (lo + hi)
        T_mid = solve_T(ed_target, mid, f_e, T_bounds)
        nB_mid = float(f_nB.ev(T_mid, mid))
        if abs(nB_mid - nB_target) < ACCURACY * max(1.0, abs(nB_target)):
            return T_mid, mid
        if nB_target < nB_mid:
            hi = mid
        else:
            lo = mid
    return T_mid, mid


def invert_tables(input_dir: Path, output_dir: Path) -> None:
    """
    1. Load raw BEST tables.
    2. Convert to physical units and build splines.
    3. Invert (e,nB)→(T,μB), evaluate P, compute cs2 and derivatives.
    4. Save P, T, μB, cs2 plus negative-cs2 points with derivatives.
    """
    # Load raw data
    ed_raw = np.loadtxt(input_dir / "BEST_EnerDens_Final.dat")[:, 2]
    nB_raw = np.loadtxt(input_dir / "BEST_BarDens_Final.dat")[:, 2]
    P_raw  = np.loadtxt(input_dir / "BEST_Press_Final.dat")[:, 2]

    # Define axes
    n_muB, n_T = 451, 771
    muB_axis = np.linspace(0.000, 0.450, n_muB)
    T_axis   = np.linspace(0.030, 0.800, n_T)

    # Reshape and scale
    ed_raw = ed_raw.reshape(n_T, n_muB) * (T_axis[:, None]**4) / (HBAR_C**3)
    P_raw  = P_raw.reshape(n_T, n_muB) * (T_axis[:, None]**4) / (HBAR_C**3)
    nB_raw = nB_raw.reshape(n_T, n_muB) * (T_axis[:, None]**3) / (HBAR_C**3)
    
    console.print(
        f"[bold cyan]Input ranges:[/]\n"
        f" e: {ed_raw.min():.3e} → {ed_raw.max():.3e} GeV/fm³\n"
        f" P: {P_raw.min():.3e} → {P_raw.max():.3e} GeV/fm³\n"
        f" nB: {nB_raw.min():.3e} → {nB_raw.max():.3e} fm⁻³"
    )

    # Build interpolants
    f_e, f_nB, f_P = build_interpolants(ed_raw, nB_raw, P_raw, T_axis, muB_axis)

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)

    for itab, (ne, nnB) in enumerate(zip(NE_LIST, NNB_LIST)):
        console.print(f"\n[bold]Table {itab}[/] – grid {ne}×{nnB}")
        ed_vals = np.linspace(ED_BOUNDS[itab], ED_BOUNDS[itab+1], ne)
        nB_vals = np.linspace(0.0, NB_BOUNDS[itab], nnB)

        # Allocate result arrays
        p_grid   = np.empty((ne, nnB))
        T_grid   = np.empty((ne, nnB))
        muB_grid = np.empty((ne, nnB))

        # Inversion loop with progress bar
        total = ne * nnB
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console, transient=True) as prog:
            task = prog.add_task("inverting", total=total)
            for i, e_val in enumerate(ed_vals):
                for j, nB_val in enumerate(nB_vals):
                    T_val, muB_val = solve_T_muB(e_val, nB_val, f_e, f_nB)
                    p_grid[i,j]   = float(f_P.ev(T_val, muB_val))
                    T_grid[i,j]   = T_val
                    muB_grid[i,j] = muB_val
                    prog.advance(task)

        # Compute cs2 grid and its derivatives
        dP_de  = np.gradient(p_grid, ed_vals, axis=0)
        dP_dnB = np.gradient(p_grid, nB_vals, axis=1)
        ratio  = nB_vals[None,:] / (ed_vals[:,None] + p_grid)
        term   = dP_dnB * ratio
        cs2    = dP_de + term

        # Write main tables
        np.savetxt(output_dir / f"BEST_eos_p_{itab}.dat",    p_grid,    fmt="%.8e")
        np.savetxt(output_dir / f"BEST_eos_T_{itab}.dat",    T_grid,    fmt="%.8e")
        np.savetxt(output_dir / f"BEST_eos_muB_{itab}.dat", muB_grid,  fmt="%.8e")
        np.savetxt(output_dir / f"BEST_eos_cs2_{itab}.dat", cs2,  fmt="%.8e")

        # Identify negative-cs2 points and save all variables there
        mask = cs2 < 0.0
        if np.any(mask):
            idx_flat = mask.flatten()
            # Flatten and mask arrays
            E_flat     = np.repeat(ed_vals, nnB)[idx_flat]
            nB_flat    = np.tile(nB_vals, ne)[idx_flat]
            P_flat     = p_grid.flatten()[idx_flat]
            T_flat     = T_grid.flatten()[idx_flat] * 1000 # Convert to MeV
            muB_flat   = muB_grid.flatten()[idx_flat] * 1000 # Convert to MeV
            dP_de_flat = dP_de.flatten()[idx_flat]
            dP_dnB_flat= dP_dnB.flatten()[idx_flat]
            ratio_flat = ratio.flatten()[idx_flat]
            term_flat  = term.flatten()[idx_flat]
            cs2_flat   = cs2.flatten()[idx_flat]
            # Stack columns: e, nB, P, T, μB, dP/de, dP/dnB, cs2
            out_arr = np.vstack([E_flat, nB_flat, P_flat, T_flat, muB_flat, dP_de_flat, dP_dnB_flat, ratio_flat, term_flat, cs2_flat]).T
            header = (
                "e [GeV/fm³]  nB [fm⁻³]      P [GeV/fm³]    T [MeV]        μB [MeV]       "
                "dP/de          dP/dnB [GeV]    nB/w [GeV⁻¹]   (dP/dnB)*nB/w   c_s²"
            )
            np.savetxt(
                output_dir / f"BEST_eos_negative_cs2_{itab}.dat",
                out_arr,
                fmt="%.8e",
                header=header
            )
            console.print(
                f"[yellow]→ Found {out_arr.shape[0]} negative-cs2 points, "
                f"saved with derivatives to BEST_eos_negative_cs2_{itab}.dat"
            )
        else:
            console.print(f"[green]✔[/] No negative-cs2 points in table {itab}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invert (T, μB) EOS tables to (e, nB) grids, compute cs2, and flag negatives with derivatives."
    )
    parser.add_argument(
        "--input",  "-i", type=Path, required=True,
        help="Directory with BEST_*_Final.dat"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Where to write inverted tables"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invert_tables(args.input, args.output)

if __name__ == "__main__":
    main()
