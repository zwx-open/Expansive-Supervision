import os


def run(use_cuda, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_cuda)  # '0,1'
    script = "python main.py " + " ".join(args)
    print(f"Running: {script}")
    os.system(script)


def run_standard_nerf(use_cuda=0):
    args = [
        "--nerf_dataset",
        "syn",
        "--nerf_backbone",
        "tensorf",
        "--strategy",
        "full",
        "--tag",
        "full",
        "--lr",
        "0.0001",
        "--up_folder_name",
        "nerf_syn",
        "--nerf_scene",
        "lego",
        # "--multi_data",
        
    ]
    run(use_cuda, args)

def run_egra_nerf(use_cuda=0):
    args = [
        "--nerf_dataset",
        "syn",
        "--nerf_backbone",
        "tensorf",
        "--strategy",
        "egra",
        "--tag",
        "egra",
        "--lr",
        "0.0001",
        "--up_folder_name",
        "nerf_syn",
        "--nerf_scene",
        "lego",
        # "--multi_data",
        
    ]
    run(use_cuda, args)

def run_soft_nerf(use_cuda=0):
    args = [
        "--nerf_dataset",
        "syn",
        "--nerf_backbone",
        "tensorf",
        "--strategy",
        "soft",
        "--tag",
        "soft_mining",
        "--lr",
        "0.0001",
        "--up_folder_name",
        "nerf_syn",
        "--nerf_scene",
        "lego",
        # "--multi_data",
        
    ]
    run(use_cuda, args)

def run_expansive_nerf(use_cuda=0):
    args = [
        "--nerf_dataset",
        "syn",
        "--nerf_backbone",
        "tensorf",
        "--strategy",
        "expansive",
        "--tag",
        "expansive",
        "--lr",
        "0.0001",
        "--up_folder_name",
        "nerf_syn",
        "--nerf_scene",
        "lego",
        # "--multi_data",
        
    ]
    run(use_cuda, args)


if __name__ == "__main__":
    run_standard_nerf(use_cuda=0)
    # run_egra_nerf(use_cuda=1)
    # run_soft_nerf(use_cuda=2)
    # run_expansive_nerf(use_cuda=3)