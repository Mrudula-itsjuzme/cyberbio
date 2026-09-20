import sys

def modify_runner(filename, is_ref=False):
    with open(filename, "r") as f:
        content = f.read()
    
    # Add --smoke argument
    if "parser.add_argument(\"--smoke\"" not in content:
        content = content.replace(
            "parser.add_argument(\"--seed\"",
            "parser.add_argument(\"--smoke\", type=str, default=\"true\", help=\"True for smoke mode\")\n    parser.add_argument(\"--seed\""
        )
    
    if not is_ref:
        # replace hardcoded paths
        content = content.replace(
            'out_dir = f"results/arena/smoke_carry_reset_{args.method}_paired_seed{args.seed}"',
            'out_dir = f"results/arena/smoke_carry_reset_{args.method}_paired_seed{args.seed}" if args.smoke.lower() == "true" else f"results/arena/carry_reset_{args.method}_paired_seed{args.seed}"'
        )
        content = content.replace(
            'summary["SMOKE_ONLY_NON_CANONICAL"] = True',
            'summary["SMOKE_ONLY_NON_CANONICAL"] = (args.smoke.lower() == "true")'
        )
        # also update reference loader
        content = content.replace(
            'ref_base = f"results/arena/carry_reset_reference_seed{args.seed}"',
            'ref_base = f"results/arena/smoke_carry_reset_reference_seed{args.seed}" if args.smoke.lower() == "true" else f"results/arena/carry_reset_reference_seed{args.seed}"'
        )
    else:
        # For build_reference_trajectory
        if "parser.add_argument(\"--smoke\"" not in content:
            content = content.replace(
                "parser.add_argument(\"--budget\"",
                "parser.add_argument(\"--smoke\", type=str, default=\"true\")\n    parser.add_argument(\"--budget\""
            )
        content = content.replace(
            'out_dir = f"results/arena/carry_reset_reference_seed{seed}"',
            'out_dir = f"results/arena/smoke_carry_reset_reference_seed{seed}" if args.smoke.lower() == "true" else f"results/arena/carry_reset_reference_seed{seed}"'
        )

    with open(filename, "w") as f:
        f.write(content)

modify_runner("scripts/run_carry_reset_pair.py")
modify_runner("scripts/build_reference_trajectory.py", is_ref=True)
