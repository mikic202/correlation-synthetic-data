from external.tab_pfn_gen.src.tabpfgen.tabpfgen import TabPFGen
from performance_metrics.measure_synthetic_data_generation_time import (
    measure_synthetic_data_generation_time,
)
import torch


if __name__ == "__main__":
    print("Running TabPFGen synthetic data generation time measurement...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    generator = TabPFGen(n_sgld_steps=100, device=device)
    time_taken = measure_synthetic_data_generation_time(
        generator.generate_classification
    )

    print("Synthetic data generation took: ", time_taken)
