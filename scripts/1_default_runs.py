from pathlib import Path

from autoverify.portfolio import Hydra, PortfolioScenario
from autoverify.util.instances import read_vnncomp_instances
from autoverify.portfolio import Hydra, PortfolioScenario
from autoverify.portfolio import ConfiguredVerifier, Portfolio, PortfolioRunner
from autoverify.util.verifiers import get_verifier_configspace


def main():
    # Run each default setting
    for network_total in [1, 3]:
        for type in ["train", "data"]:

            dataset_name = (
                f"mnist_fc_single_{type}" if network_total == 1 else f"mnist_fc_{type}"
            )

            # Read dataset
            dataset = read_vnncomp_instances(
                dataset_name, vnncomp_path=Path("../vnncomp/vnncomp2022/benchmarks")
            )

            pf_default = Portfolio.from_json(
                get_verifier_configspace("nnenum").get_default_configuration()
            )
            pf_runner = PortfolioRunner(pf_default)
            pf_runner.verify_instances(
                dataset,
                out_json=Path(
                    f"results/default/runs/results_default_net_{network_total}_{type}_nnenum.json"
                ),
            )

    return 0


if __name__ == "__main__":
    main()
