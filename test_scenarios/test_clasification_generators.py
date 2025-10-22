from external.tab_pfn_gen.src.tabpfgen.tabpfgen import TabPFGenClassifier
from model_wrappers.full_tabpfn_gen import FullTabpfnGen
from model_wrappers.smote_generator import SmoteGenerator
from model_wrappers.ctgan_generator import CTGANGenerator

import torch
from performance_metrics.measure_model_performance import (
    measure_model_clasification_performance,
)
from time import time


N_SMAPLES = 100


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure performance of TabPFGen with different n_sgld_steps"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Which model will be run",
    )
    parser.add_argument(
        "--n_sgld_steps",
        type=int,
        default=100,
        help="Number of SGLD steps to use in TabPFGen",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=N_SMAPLES,
        help="Number of samples to generate for performance measurement",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results to, defaults to <model_name>.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.output is None:
        args.output = f"{args.model}.csv"

    if args.model.lower() == "tabpfgen":
        generator = TabPFGenClassifier(n_sgld_steps=100, device=device)
    elif args.model.lower() == "fulltabpfgen":
        generator = FullTabpfnGen(str(device))
    elif args.model.lower() == "smote":
        generator = SmoteGenerator()
    elif args.model.lower() == "ctgan":
        generator = CTGANGenerator()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    results = measure_model_clasification_performance(
        generator.generate_classification, 1, n_samples=args.n_samples
    )
    results.to_csv(args.output)
    print(f"Test finished. Results saved to {args.output}")
