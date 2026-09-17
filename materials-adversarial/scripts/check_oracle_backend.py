import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hpc_oracle.backend import QuantumEspressoBackend, VASPBackend

def main():
    parser = argparse.ArgumentParser(description="Check Oracle Backend Readiness")
    parser.add_argument("--backend", type=str, required=True, choices=["qe", "vasp"], help="Backend to check")
    parser.add_argument("--config", type=str, default="hpc_oracle/configs/default.json", help="Path to config")
    args = parser.parse_args()
    
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            
    if args.backend == "qe":
        backend = QuantumEspressoBackend(config)
    else:
        backend = VASPBackend(config)
        
    status = backend.validate_environment()
    
    print(f"Backend Check: {args.backend.upper()}")
    print("-" * 40)
    print(f"Available: {status.available}")
    print(f"Executable: {status.executable_path or 'NOT FOUND'}")
    print(f"Pseudopotential Path: {status.pseudopotential_path or 'NOT FOUND'}")
    
    if not status.available:
        print("\\nScientific Blockers / Missing Requirements:")
        for req in status.missing_requirements:
            print(f" - {req}")
        sys.exit(1)
        
    print("\\nEnvironment is ready.")
    sys.exit(0)

if __name__ == "__main__":
    main()
