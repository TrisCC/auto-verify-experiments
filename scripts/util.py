from autoverify.util.instances import read_verification_result_from_json, json_write_verification_result
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path


def split_mix():
    data = read_verification_result_from_json("./grace_dump7/default/results_mix.json")
    
    data_mnist = []
    data_tllverifybench = []
    
    for result in data:
        if "mnist" in result.network:
            data_mnist.append(result)
        elif "tllverifybench" in result.network:
            data_tllverifybench.append(result)
    for result in data_mnist:
        json_write_verification_result(result, Path("./grace_dump7/default/results_mnist_fc_verinet_nnenum.json"))
        
    for result in data_tllverifybench:
        json_write_verification_result(result, Path("./grace_dump7/default/results_tllverifybench_verinet_nnenum.json"))
    
    return 0

def avg_runtime():
    data = read_verification_result_from_json("./grace_dump7/default/results_mnist_fc_nnenum.json")
    res = {
        "2": [],
        "4": [],
        "6": []
    }
    
    for instance in data:
        if instance.network.endswith("2.onnx"):
            res["2"].append(float(instance.took))
        elif instance.network.endswith("4.onnx"):
            res["4"].append(float(instance.took))
        elif instance.network.endswith("6.onnx"):
            res["6"].append(float(instance.took))
            
    for num in ["2", "4", "6"]:
        res[f"{num}_mean"] = np.mean(res[num])
        res[f"{num}_std"] = np.std(res[num])
        
    return res

if __name__ == '__main__':
    print(avg_runtime())
    