
import torch
from pathlib import Path
import sys

# Ensure project root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from experiments.run_3d import run_3d_experiment
from src.neural_pbf.physics.material import MaterialConfig
from src.neural_pbf.core.config import SimulationConfig, LengthUnit

def run_compare():
    # 1. High-Fidelity Configuration (Higher Resolution & Real Conductivity)
    # ------------------------------------------------------------------
    # We increase Nx, Ny, Nz for resolution
    # and set k_solid/k_liquid to higher, more physical values
    mat_cfg_hi = MaterialConfig(
        k_powder=0.2,      
        k_solid=25.0,      # Higher conductivity (Stainless Steel / Ti64 range)
        k_liquid=45.0,     # Higher conductivity
        cp_base=500.0,     
        rho=7900.0,        
        T_solidus=1650.0,  
        T_liquidus=1700.0, 
        latent_heat_L=2.7e5, 
        transition_sharpness=5.0
    )
    
    sim_cfg_hi = SimulationConfig(
        Lx=1.0, Ly=0.5, Lz=0.25,
        Nx=512, Ny=256, Nz=128,
        dt_base=2e-6,
        length_unit=LengthUnit.MILLIMETERS
    )

    print("\n>>> STARTING HIGH-FIDELITY RUN (Higher Resolution + Conductivity) <<<")
    run_3d_experiment(
        run_name="compare_high_fid",
        mat_cfg=mat_cfg_hi,
        sim_cfg=sim_cfg_hi,
        total_time=0.4e-3
    )

    # 2. Low-Fidelity Configuration (Base / Prototype)
    # ----------------------------------------------
    mat_cfg_lo = MaterialConfig(
        k_powder=0.2,      
        k_solid=15.0,      
        k_liquid=30.0,     
        cp_base=500.0,     
        rho=7900.0,        
        T_solidus=1650.0,  
        T_liquidus=1700.0, 
        latent_heat_L=2.7e5, 
        transition_sharpness=5.0
    )
    
    sim_cfg_lo = SimulationConfig(
        Lx=0.8, Ly=0.4, Lz=0.2, 
        Nx=40, Ny=20, Nz=10,    # Very low resolution for fast prototyping
        dt_base=2e-6,
        length_unit=LengthUnit.MILLIMETERS
    )

    print("\n>>> STARTING LOW-FIDELITY RUN (Base Prototype) <<<")
    run_3d_experiment(
        run_name="compare_low_fid",
        mat_cfg=mat_cfg_lo,
        sim_cfg=sim_cfg_lo,
        total_time=0.4e-3
    )

if __name__ == "__main__":
    run_compare()
