from external.tab_pfn_gen.src.tabpfgen.tabpfgen import TabPFGen
from performance_metrics.measure_privacy import measure_privacy
from test_datasets.dataset_getters import get_climate_model_simulation_dataset
import torch
from model_wrappers.full_tabpfn_gen import FullTabpfnGen


if __name__ == "__main__":
    print("Running TabPFGen synthetic data generation time measurement...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    generator = TabPFGen(device=str(device))

    measure_privacy(
        generator.generate_classification, get_climate_model_simulation_dataset
    )
