#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={cpus}
#SBATCH --mem={memory}
#SBATCH --time={wall_time}

module load {modules}
cd {scratch_dir}

{executable} < {input_file} > {output_file}
