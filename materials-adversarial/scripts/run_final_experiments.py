#!/usr/bin/env python3
import argparse
import sys
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Reproducibility Script for Final Project")
    parser.add_argument("--smoke", action="store_true", help="Quick end-to-end validation")
    parser.add_argument("--analysis", action="store_true", help="Rebuild analysis from artifacts")
    parser.add_argument("--multiseed", action="store_true", help="Run central multi-seed experiments")
    parser.add_argument("--full", action="store_true", help="Run everything scientifically necessary")
    args = parser.parse_args()

    if not any([args.smoke, args.analysis, args.multiseed, args.full]):
        parser.print_help()
        sys.exit(1)

    if args.smoke or args.full:
        print("\n--- SMOKE TEST ---")
        run_cmd("PYTHONPATH=. python scripts/smoke_test_attack.py")

    if args.multiseed or args.full:
        print("\n--- MULTI-SEED VALIDATION ---")
        run_cmd("PYTHONPATH=. python scripts/run_multiseed.py")

    if args.analysis or args.full:
        print("\n--- ANALYSIS REBUILD ---")
        # Ensure tests pass as part of full reproducible check
        run_cmd("PYTHONPATH=. pytest tests -q")
        print("Analysis generated during multi-seed run and stored in results/final/")

if __name__ == "__main__":
    main()
