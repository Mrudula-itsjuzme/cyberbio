import os
from typing import Dict, Any, List

def generate_slurm_script(
    job_name: str,
    output_dir: str,
    executable_path: str,
    input_file: str,
    output_file: str,
    nodes: int = 1,
    ntasks: int = 1,
    cpus_per_task: int = 1,
    memory: str = "4G",
    wall_time: str = "24:00:00",
    partition: str = "standard",
    modules: List[str] = None,
    extra_srun_args: str = "",
    array_size: int = None
) -> str:
    """
    Generate portable SLURM scripts.
    """
    script = []
    script.append("#!/bin/bash")
    script.append(f"#SBATCH --job-name={job_name}")
    script.append(f"#SBATCH --nodes={nodes}")
    script.append(f"#SBATCH --ntasks={ntasks}")
    script.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    script.append(f"#SBATCH --mem={memory}")
    script.append(f"#SBATCH --time={wall_time}")
    script.append(f"#SBATCH --partition={partition}")
    
    if array_size is not None:
        script.append(f"#SBATCH --array=1-{array_size}")
        
    script.append(f"#SBATCH --output={output_dir}/%x_%A_%a.out" if array_size else f"#SBATCH --output={output_dir}/%x_%j.out")
    script.append(f"#SBATCH --error={output_dir}/%x_%A_%a.err" if array_size else f"#SBATCH --error={output_dir}/%x_%j.err")
    script.append("")
    
    if modules:
        for mod in modules:
            script.append(f"module load {mod}")
    
    script.append("")
    script.append(f"cd {output_dir}")
    
    # We do not hardcode scratch paths, the user can provide wrapper scripts in executable_path
    script.append(f"srun {extra_srun_args} {executable_path} < {input_file} > {output_file}")
    script.append("")
    
    return "\\n".join(script)
