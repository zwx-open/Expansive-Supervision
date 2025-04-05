import os

from util import misc
from util.tensorboard import writer
from util.logger import log
from util.recorder import recorder
from opt import Opt

from trainer.nerf_trainer import NeRFTrainer
import copy
import torch

DatasetSetting = {
    "syn": {
        "data": "nerf_synthetic",
        "scene_list": [
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
        ],
    },
    "nsvf": {
        "data": "Synthetic_NSVF",
        "scene_list": [
            "Bike",
            "Lifestyle",
            "Palace",
            "Robot",
            "Spaceship",
            "Steamtrain",
            "Toad",
            "Wineholder",
        ],
    },
    "llff": {
        "data": "nerf_llff_data",
        "scene_list": [
            "fern",
            "flower",
            "room",
            "leaves",
            "horns",
            "trex",
            "fortress",
            "orchids",
        ],
    },
    "tt": {
        "data": "TanksAndTemple",
        "scene_list": ["Barn", "Caterpillar", "Family", "Ignatius", "Truck"],
    },
    "mip360": {
        "data": "mip-nerf-360",
        "scene_list": [
            "bicycle",
            "bonsai",
            "counter",
            "garden",
            "kitchen",
            "room",
            "stump",
        ],
    },
}


def main():
    opt = Opt()
    args = opt.get_args()

    misc.fix_seed(args.seed)
    log.inst.success("start")
    writer.init_path(args.save_folder)
    log.set_export(args.save_folder)

    #########################
    if args.multi_data:
        dataset = args.nerf_dataset
        samples = DatasetSetting[dataset]["scene_list"]
        process_task(samples, args, cuda_num=0)
    else:
        start_trainer(args)

    #########################
    time_dic = log.end_all_timer()

    table = recorder.add_main_table()
    if table:
        # recorder.add_summary_table()
        recorder.dic["time"] = time_dic
        recorder.add_time_table()
        recorder.dump_table(os.path.join(args.save_folder, "res_table.md"))

    writer.close()
    log.inst.success("Done")


def process_task(sample_list, args, cuda_num=0):
    torch.set_num_threads(16)
    results = []
    for sample in sample_list:
        cur_args = copy.deepcopy(args)
        cur_args.device = f"cuda:{cuda_num}"
        cur_args.nerf_scene = sample
        cur_res = start_trainer(cur_args)
        results.append(cur_res)
    return results


def start_trainer(args):
    if args.signal_type == "radiance_field":
        trainer = NeRFTrainer(args)
    else:
        raise NotImplementedError
    trainer.train()
    res = getattr(trainer, "result", None)
    return res


if __name__ == "__main__":
    main()
