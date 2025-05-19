import pandas as pd
from pandas import DataFrame
import os


def process_results(results_dir):
    data = pd.DataFrame(
        {
            "Num. of NNENUM configurations": [1],
            "Num. of problem instances used for training": [0],
            "Average wall-clock time (s)": [47],
            "Num. of timeouts": [143],
            "PAR10 score (s)": [339],
        }
    )

    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(results_dir, filename)
            df = pd.read_csv(filepath)

            # Extract relevant information from filename
            parts = filename.replace(".csv", "").split("_")
            if (
                len(parts) == 5
            ):  # Expected format: approach_dataset_split_seed (no timestamp)
                _, total_pf, _, total_nets, _ = parts
            else:
                print(f"Skipping file with unexpected format: {filename}")
                continue

            if total_nets == "1":
                continue

            for index, row in df.iterrows():
                data = pd.concat(
                    [
                        data,
                        pd.DataFrame(
                            {
                                "Num. of NNENUM configurations": [total_pf],
                                "Num. of problem instances used for training": [
                                    round(row["target_instances_optimized"])
                                ],
                                "Average wall-clock time (s)": [
                                    round(row["target_mean"])
                                ],
                                "Num. of timeouts": [round(row["target_timeouts"])],
                                "PAR10 score (s)": [round(row["target_par10"])],
                            }
                        ),
                    ],
                    join="inner",
                )

    return data


if __name__ == "__main__":
    results_directory = "./results"  # Replace with your actual directory

    df_results = process_results(results_directory)

    output_file = "results_table.tex"
    output_csv = "results.csv"

    df_results = df_results.sort_values(
        [
            "Num. of NNENUM configurations",
            "Num. of problem instances used for training",
        ],
        ascending=[True, True],
    )

    # Save DataFrame to CSV
    df_results.to_csv(output_csv, index=False)

    # Convert DataFrame to LaTeX
    latex_output = df_results.to_latex(index=False, escape=False)

    # Save LaTeX output to a file
    with open(output_file, "w") as f:
        f.write(latex_output)

    print(f"LaTeX table generated and saved to {output_file}")
