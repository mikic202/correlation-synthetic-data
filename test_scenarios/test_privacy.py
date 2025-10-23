from external.tab_pfn_gen.src.tabpfgen.tabpfgen import TabPFGen, TabPFGenClassifier
from performance_metrics.measure_privacy import measure_privacy
from test_datasets.dataset_getters import (
    get_climate_model_simulation_dataset,
    get_wdbc_dataset,
)
import torch
from model_wrappers.full_tabpfn_gen import FullTabpfnGen


if __name__ == "__main__":
    print("Running TabPFGen synthetic data generation time measurement...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    generator = TabPFGenClassifier(device=str(device))
    full_generator = FullTabpfnGen(device=str(device))

    print(
        measure_privacy(
            [generator, full_generator],
            get_wdbc_dataset,
            1000,
        )
    )
