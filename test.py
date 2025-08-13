import torch
from performance_metrics.measure_model_performance import (
    measure_model_clasification_performance,
)
from time import time
from model_wrappers.full_tabpfn_gen import FullTabpfnGen
from feature_order.corelation_based_order import (
    generate_correlation_based_order_of_features,
)
from feature_order.graph_based_order import generate_graph_based_order_of_features
from feature_order.tree_based_order import (
    generate_tree_based_order_of_features,
    generate_xgboost_based_order_of_features,
)

ordering_types = {
    "corelation": generate_correlation_based_order_of_features,
    "graph": generate_graph_based_order_of_features,
    "tree": generate_tree_based_order_of_features,
    "xgb": generate_xgboost_based_order_of_features,
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    start = time()

    for ordering_type in ordering_types:
        generator = FullTabpfnGen(str(device), ordering_types[ordering_type])
        results = measure_model_clasification_performance(
            generator, 1, n_samples=100
        ).to_csv("ordering_type.csv")
        print(f"Results for {ordering_type} based order")
        print(results)
        print("_____________________________________")
