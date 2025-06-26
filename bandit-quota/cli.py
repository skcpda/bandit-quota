import argparse, sys
from .core import run_experiment, list_datasets

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="bandit-quota")
    ap.add_argument("--dataset", default="scifact",
                    choices=list_datasets(),
                    help="BEIR dataset key (default: scifact)")
    ap.add_argument("--policy", default="bandit",
                    choices=["bandit", "union"],
                    help="'bandit' or 'union' baseline")
    args = ap.parse_args(argv)
    run_experiment(dataset=args.dataset, policy=args.policy)

if __name__ == "__main__":                 # `python -m bandit_quota`
    main(sys.argv[1:])
