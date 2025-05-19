from autoverify.util.instances import (
    read_verification_result_from_json,
    json_write_verification_result,
    VerificationResultString,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import csv
from pathlib import Path
from os import listdir
from os.path import isfile, join


def main():
    limit_results = 1000
    # list_portfolio_len = [1, 2, 3]
    list_portfolio_len = [2]
    list_num_instances = [8, 16, 32, 64, 128, 256, 512]
    # list_num_instances = [8, 16, 32, 64]
    
    
    for portfolio_len in list_portfolio_len:
        
        stats = {}
        for num_instances in list_num_instances:
            plot_data(
                f"mnist_fc_24h/24hour_training_mnist_fc_{portfolio_len}/runs/results_{num_instances}_instances.json",
                "mnist_fc_24h/default/results_mnist_fc_data_nnenum.json",
                "NNENUM",
                f"mnist_fc_24h/24hour_training_mnist_fc_{portfolio_len}/runs/",
                portfolio_len,
                num_instances,
                1000,  
                f"mnist_fc_24h/24hour_training_mnist_fc_{portfolio_len}/runs/results_{num_instances}_train_instances.json",
                "mnist_fc_24h/default/results_mnist_fc_train_nnenum.json",
            )

            # stats[num_instances] = calc_stats(
            #     f"mnist_fc_24h/24hour_training_mnist_fc_{portfolio_len}/runs/results_{num_instances}_instances.json",
            #     "mnist_fc_24h/default/results_mnist_fc_data_nnenum.json",
            #     f"mnist_fc_24h/24hour_training_mnist_fc_{portfolio_len}/runs/delta_{num_instances}_nnenum.json",
            #     num_instances,
            #     1000
            # )
        
        # with open(f"mnist_fc_24h/24hour_training_mnist_fc_{portfolio_len}_stats.csv", "w") as f:
        #     w = csv.DictWriter(f, stats[64].keys())
        #     w.writeheader()
        #     for num_instances in list_num_instances:
        #         w.writerow(stats[num_instances])


    return 0

def main2():
    list_num_instances = [8, 16, 32, 64, 128, 256, 512]
    stats = {}
    
    for num_instances in list_num_instances:
        plot_data(
                f"mnist_fc_24h/24hour_training_mnist_fc_single/runs/results_{num_instances}_instances.json",
                "mnist_fc_24h/default/results_mnist_fc_single_data_nnenum.json",
                "NNENUM",
                f"mnist_fc_24h/24hour_training_mnist_fc_single/runs/",
                2,
                num_instances,
                1000,  
                f"mnist_fc_24h/24hour_training_mnist_fc_single/runs/results_{num_instances}_train_instances.json",
                "mnist_fc_24h/default/results_mnist_fc_single_train_nnenum.json",
            )
        
        stats[num_instances] = calc_stats(
                f"mnist_fc_24h/24hour_training_mnist_fc_single/runs/results_{num_instances}_instances.json",
                "mnist_fc_24h/default/results_mnist_fc_single_data_nnenum.json",
                f"mnist_fc_24h/24hour_training_mnist_fc_single/runs/delta_{num_instances}_nnenum.json",
                num_instances,
                1000
            )
        
    with open(f"mnist_fc_24h/24hour_training_mnist_fc_single_stats.csv", "w") as f:
        w = csv.DictWriter(f, stats[64].keys())
        w.writeheader()
        for num_instances in list_num_instances:
            w.writerow(stats[num_instances])
    
    return 0

def main3():
    iters = [{"PF": 1,"NET":1},
             {"PF": 1,"NET":3},
             {"PF": 2,"NET":1},
             {"PF": 2,"NET":3},
             {"PF": 3,"NET":3},
             ]
    
    for i in iters:
        stats = {}
        num_instances_list = []
        path = f"mnist_fc_24h/pf_{i['PF']}_net_{i['NET']}/runs"
        base_files = [f for f in listdir(path) if isfile(join(path, f)) and f.startswith("results") and f.find("train") == -1]
        
        for file in base_files:
            num_instances = file.split("_")[1]
            
            try:
                if(len([f for f in listdir(path) if isfile(join(path, f)) and f == f"results_{num_instances}_train_instances.json"]) == 1):
                    plot_data(path + "/" + file,
                            f"mnist_fc_24h/default/results_default_net_{i['NET']}_data_nnenum.json",
                            "NNENUM",
                            path + "/",
                            i["PF"],
                            int(num_instances),
                            1000,
                            path + "/" + f"results_{num_instances}_train_instances.json",
                            f"mnist_fc_24h/default/results_default_net_{i['NET']}_train_nnenum.json")
                else:
                    plot_data(path + "/" + file,
                            f"mnist_fc_24h/default/results_default_net_{i['NET']}_data_nnenum.json",
                            "NNENUM",
                            path + "/",
                            i["PF"],
                            int(num_instances),
                            1000,
                            None,
                            None)
                    
                stats[int(num_instances)] = calc_stats(
                    path + "/" + file,
                    f"mnist_fc_24h/default/results_default_net_{i['NET']}_data_nnenum.json",
                    f"{path}/delta_{num_instances}_nnenum.json",
                    
                    num_instances,
                    1000
                )
                num_instances_list.append(int(num_instances))
            except:
                continue
        
        with open(f"mnist_fc_24h/pf_{i['PF']}_net_{i['NET']}_stats.csv", "w") as f:
            w = csv.DictWriter(f, stats[64].keys())
            w.writeheader()
            for num_instances in num_instances_list:
                w.writerow(stats[num_instances])
                    
        
        

def calc_par10(data1, alg_name, folder):
    results = {}
    data = read_verification_result_from_json(folder + data1)

    par10_took = 0
    for instance in data:
        if float(instance.took) >= 300:
            par10_took += 3000
        else:
            par10_took += float(instance.took)
    par10_took /= len(data)

    results = {
        "total_took": np.sum([float(item.took) for item in data]),
        "mean_took": np.mean([float(item.took) for item in data]),
        "par10_took": par10_took,
    }

    return results


def calc_stats(optimized_results_path, default_results_path, dest_file, num_instances, limit_results):
    optimized_results = read_verification_result_from_json(
        optimized_results_path
    )
    default_results = read_verification_result_from_json(default_results_path)
    total_optimized_results = len(optimized_results)
    total_default_results = len(default_results)


    sorted_lists = find_matching_elements(optimized_results, default_results)
    optimized_results = sorted_lists[0][:limit_results]
    default_results = sorted_lists[1][:limit_results]

    # Extract data for plotting
    x = [float(item.took) for item in optimized_results]
    y = [float(item.took) for item in default_results]
    delta = []
    for idx, number in enumerate(x):
        delta.append(number - y[idx])

    par10_took_target = 0
    for instance in optimized_results:
        if float(instance.took) >= 300:
            par10_took_target += 3000
        else:
            par10_took_target += float(instance.took)
    par10_took_target /= len(optimized_results)

    par10_took_default = 0
    for instance in default_results:
        if float(instance.took) >= 300:
            par10_took_default += 3000
        else:
            par10_took_default += float(instance.took)
    par10_took_default /= len(optimized_results)
    
    res = {
                "target_instances_optimized": num_instances,
                "target_total_verified": total_optimized_results,
                "target_results_path": optimized_results_path,
                "target_instances": len(optimized_results),
                "target_success": sum(
                    1
                    for i in optimized_results
                    if i.result == "UNSAT" or i.result == "SAT"
                ),
                "target_errors": sum(
                    1 for i in optimized_results if i.result == "ERR"
                ),
                "target_timeouts": sum(
                    1 for i in optimized_results if i.result == "TIMEOUT"
                ),
                "target_mean": np.mean(x),
                "target_std": np.std(x),
                "target_par10": par10_took_target,
                "default_total_verified": total_default_results,
                "default_results_path": default_results_path,
                "default_instances": len(default_results),
                "default_success": sum(
                    1
                    for i in default_results
                    if i.result == "UNSAT" or i.result == "SAT"
                ),
                "default_errors": sum(
                    1 for i in default_results if i.result == "ERR"
                ),
                "default_timeout": sum(
                    1 for i in default_results if i.result == "TIMEOUT"
                ),
                "default_mean": np.mean(y),
                "default_std": np.std(y),
                "default_par10": par10_took_default,
                "delta_total_verified": len(optimized_results),
                "delta_mean": np.mean(delta),
                "delta_std": np.std(delta),
                "delta_sum": np.sum(delta),
                "delta_max": np.max(delta),
                "delta_min": np.min(delta),
            }

    with open(dest_file, "w") as f:
        json.dump(res,f,)

    return res


def find_matching_elements(list1, list2):
    matches_1 = []
    matches_2 = []
    for elem1 in list1:
        for elem2 in list2:
            if (
                elem1.network == elem2.network
                and elem1.property == elem2.property
            ):
                matches_1.append(elem1)
                matches_2.append(elem2)
    
    to_pop = []
    for index, elem in enumerate(matches_1):
        if index > len(matches_1) - 2:
            break
        if (elem.network == matches_1[index + 1].network and elem.property == matches_2[index + 1].property):
            if (elem.took < matches_1[index + 1].took):
                to_pop.append(index + 1)
            else:
                to_pop.append(index)

    to_pop.reverse()
    for index in to_pop:
        matches_1.pop(index)
        matches_2.pop(index)
    
    return [matches_1, matches_2]


def plot_data(
    optimized_results_path,
    default_results_path,
    alg_name,
    dest_folder,
    portfolio_length,
    num_instances,
    limit_results,
    train_optimized_results_path,
    train_default_results_path,
):
    optimized_results = read_verification_result_from_json(
        optimized_results_path
    )
    default_results = read_verification_result_from_json(default_results_path)
    sorted_lists = find_matching_elements(optimized_results, default_results)
    optimized_results = sorted_lists[0][:limit_results]
    default_results = sorted_lists[1][:limit_results]

    for index, inst in enumerate(optimized_results):
        if (
            inst.network + inst.property
            != default_results[index].network + default_results[index].property
        ):
            raise ValueError(f"Networks do not match at index {index}")

    # Extract data for plotting
    x = [float(item.took) for item in optimized_results]
    y = [float(item.took) for item in default_results]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.axis([1, 10000, 1, 100000])
    ax1.loglog()

    # Configure the plot for logarithmic scale on both axes
    # plt.figure(figsize=(8, 6))
    ax1.set_xscale("log", base=10)
    ax1.set_yscale("log", base=10)

    # Create the scatterplot with different colors for each list
    ax1.scatter(x, y, s=2, c="blue", label="test set")
    ax1.axline((0, 0), (10, 10), color="grey", linestyle=":")
    
    if train_default_results_path != None:
        train_optimized_results = read_verification_result_from_json(train_optimized_results_path)
        train_default_results = read_verification_result_from_json(train_default_results_path)
        sorted_lists = find_matching_elements(train_optimized_results, train_default_results)
        train_optimized_results = sorted_lists[0][:limit_results]
        train_default_results = sorted_lists[1][:limit_results]
        
        # Extract data for plotting
        x_train = [float(item.took) for item in train_optimized_results]
        y_train = [float(item.took) for item in train_default_results]
        
        ax1.scatter(x_train, y_train, s=2, c="red", label="training set") 
    
    ax1.set_xlim(0, 300)
    ax1.set_ylim(0, 300)

    # Add labels and title
    ax1.set_xlabel("Walltime with an optimized configuration (s)")
    ax1.set_ylabel("Walltime with the default configuration (s)")
    ax1.set_title(
        f"Performance of a portfolio of {portfolio_length} instances of {alg_name} on MNIST instances tuned with {num_instances} training instances",
        wrap=True,
    )
    ax1.legend(loc="upper left")

    # Show the plot
    ax1.grid(False)
    # plt.tight_layout()
    plt.savefig(
        dest_folder
        + f"performance_{alg_name}_{portfolio_length}_{num_instances}.png"
    )

    plt.clf()


if __name__ == "__main__":
    # main()
    main3()
