import os
from manager import ParamManager
import subprocess
import numpy as np
import math


def run_single(exp_num):
    pm = ParamManager(idx=0, exp_num=exp_num)
    pm.p.up_folder_name = "debug"
    pm.p.num_epochs = 500
    pm.p.log_epoch = 50
    # test dataset
    # pm.p.input_path = "/home/wxzhang/projects/coding4paper/data"
    # pm.p.multi_data = None

    # pm.p.transform = "sym_power"
    pm.p.signal_type = "series"
    pm.p.hidden_layers = 1
    pm.p.hidden_features = 64

    use_cuda = 0
    cmd_str = pm.export_cmd_str(use_cuda=[use_cuda])
    print(f"Running: {cmd_str}")
    os.system(cmd_str)


def run_subprocess(idx_list, gpu_list, exp_num):
    processes = []

    # assert len(idx_list) == len(gpu_list)
    _len = min(len(idx_list), len(gpu_list))
    idx_list = idx_list[:_len]
    gpu_list = gpu_list[:_len]

    for idx, use_cuda in zip(idx_list, gpu_list):
        pm = ParamManager(idx=idx, exp_num=exp_num)
        cmd_str = pm.export_cmd_str(use_cuda=[use_cuda])
        ##  print cmd str for debugger
        # exit()
        process = subprocess.Popen(cmd_str, shell=True)
        print(f"PID: {process.pid}")
        processes.append(process)

    for process in processes:
        process.wait()


def run_tasks(exp_num, param_idxs, gpu_list):

    # param_idxs:list = [0,1,2,3,4]

    gpus = len(gpu_list)
    rounds = math.ceil(len(param_idxs) / gpus)
    print("rounds: ", rounds)

    for i in range(rounds):
        cur_param_idxs = param_idxs[i * gpus : min(len(param_idxs), (i + 1) * gpus)]
        cur_len = len(param_idxs)
        gpu_list = gpu_list[:cur_len]
        run_subprocess(cur_param_idxs, gpu_list, exp_num)


if __name__ == "__main__":
    
    param_idxs = [
                # "soft",
                  "full",
                 #  "expansive","egra"
                  ]
    gpu_list = [0,1,2,3]

    run_tasks("060", param_idxs, gpu_list)
