import sys
import os

from pathlib import Path

from autoverify.portfolio import Hydra, PortfolioScenario
from autoverify.util.instances import read_vnncomp_instances
from autoverify.portfolio import Hydra, PortfolioScenario
from autoverify.portfolio import ConfiguredVerifier, Portfolio, PortfolioRunner
from autoverify.util.verifiers import get_verifier_configspace


def main():
    num_instances = 1

    try:
        num_instances = int(sys.argv[1])
    except (IndexError, ValueError):
        raise ValueError(
            "Expected num_instances (int) to be given as command line argument"
        )

    dataset_data = read_vnncomp_instances(
        "mnist_fc_single_data", vnncomp_path=Path("../vnncomp/vnncomp2022/benchmarks")
    )

    print(f"Running portfolio with {num_instances}")
    run_verifier(
        f"{str(num_instances)}_mnist_fc",
        f"results/24hour_training_mnist_fc_single/portfolios/PF_{str(num_instances)}_mnist_fc.json",
        "mnist_fc",
        num_instances,
        dataset_data,
    )

    return 0


def run_verifier(portfolio_name, portfolio_path, dataset_name, dataset_cutoff, dataset):

    # Run portfolio trained
    pf_trained = Portfolio.from_json(Path(portfolio_path))
    pf_runner = PortfolioRunner(pf_trained)
    pf_runner.verify_instances(
        dataset,
        out_json=Path(
            f"results/24hour_training_mnist_fc_single/runs/results_{str(dataset_cutoff)}_instances.json"
        ),
    )


if __name__ == "__main__":
    main()
