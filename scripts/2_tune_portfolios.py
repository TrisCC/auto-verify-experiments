import sys
import os

from pathlib import Path

from autoverify.portfolio import Hydra, PortfolioScenario
from autoverify.util.instances import read_vnncomp_instances
from autoverify.portfolio import Hydra, PortfolioScenario
from autoverify.portfolio import ConfiguredVerifier, Portfolio, PortfolioRunner
from autoverify.util.verifiers import get_verifier_configspace


def main():
    num_tune_instances = -1
    num_nnenum_instances = -1
    num_network_types = [1, 3]

    try:
        num_tune_instances = int(sys.argv[1])
        num_nnenum_instances = int(sys.argv[2])
    except (IndexError, ValueError):
        raise ValueError(
            "Expected num_tune_instances (int) and num_nnenum_instances (int) to be given as command line argument"
        )

    dataset_train = read_vnncomp_instances(
        "mnist_fc_single_train", vnncomp_path=Path("../vnncomp/vnncomp2022/benchmarks")
    )

    print(f"Training portfolio with {num_tune_instances}")
    train_portfolio(
        ["nnenum", "nnenum"],
        0.95,
        [("nnenum", 0, 0)],
        "mnist_fc_single_train",
        num_tune_instances,
        dataset_train,
    )

    return 0


def train_portfolio(
    verifiers_list, alpha, resources_list, dataset_name, dataset_cutoff, benchmark
):
    benchmark = benchmark[:: int(512 / dataset_cutoff)]
    
    nnenum_instances = len(verifiers_list)

    pf_scenario = PortfolioScenario(
        verifiers_list,
        resources_list,
        benchmark,
        len(verifiers_list),
        (60 * 60) * 24 / len(verifiers_list),
        alpha=alpha,
    )

    hydra = Hydra(pf_scenario)
    pf = hydra.tune_portfolio()
    pf.to_json(
        Path(
            f"results/pf_{nnenum_instances}_net_1/portfolios/PF_{str(dataset_cutoff)_mnist_fc.json}"
        )
    )


if __name__ == "__main__":
    main()
