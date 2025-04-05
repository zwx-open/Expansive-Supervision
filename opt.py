import argparse
import os
import json

from util.misc import gen_cur_time,gen_random_str


class Opt(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_base()
        self.add_nerf()

    def add_base(self):
        self.parser.add_argument("--seed", type=int, default=3407)
        self.parser.add_argument("--save_dir", type=str, default="./log")
        self.parser.add_argument("--tag", type=str, default="exp")
        self.parser.add_argument("--up_folder_name", type=str, default="debug")
        self.parser.add_argument("--debug", action="store_true")
        self.parser.add_argument("--device", type=str, default="cuda:0")

        # log
        self.parser.add_argument(
            "--log_epoch",
            type=int,
            default=500,
            help="log performance during training",
        )
        self.parser.add_argument(
            "--snap_epoch",
            type=int,
            default=5000,
            help="save fitted data during training",
        )

        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=4096,
            help="for radiance field which requires batch training",
        )

        # data folder
        self.parser.add_argument(
            "--data_folder",
            type=str,
            default="./data")

    def add_nerf(self):
        self.parser.add_argument(
            "--signal_type",
            type=str,
            default="radiance_field",
        )

        self.parser.add_argument(
            "--strategy",
            type=str,
            choices=["full", "soft", "expansive", "egra"],
            default="full"
        )
        self.parser.add_argument("--multi_data", action="store_true")

        self.parser.add_argument(
            "--nerf_backbone",
            type=str,
            choices=["tensorf","nerf","ingp"],
            default="tensorf")

        self.parser.add_argument(
            "--nerf_dataset",
            type=str,
            choices=["syn","nsvf","llff","mip360","tt"],
            default="syn")
        
        self.parser.add_argument(
            "--nerf_scene",
            type=str,
            default="lego")

        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        
        self.parser.add_argument(
            "--use_mse", action="store_true", help="expansive"
        )
        self.parser.add_argument(
            "--a_s_add", action="store_true", help="expansive"
        )

        # soft mining
        self.parser.add_argument(
            "--soft_mining_alpha",
            type=float,
            default=0.6,
            help="see paper Accelerating Neural Field Training via Soft Mining",
        )

        self.parser.add_argument(
            "--soft_mining_a",
            type=float,
            default=2e1,
            help="see paper Accelerating Neural Field Training via Soft Mining",
        )

        self.parser.add_argument(
            "--soft_mining_b",
            type=float,
            default=2e-2,
            help="see paper Accelerating Neural Field Training via Soft Mining",
        )
        
        self.parser.add_argument(
            "--soft_mining_warmup",
            type=int,
            default=1000,
            help="see paper Accelerating Neural Field Training via Soft Mining",
        )
        
        self.parser.add_argument(
            "--wo_correction_loss",
            action="store_true"
        )
        
        self.parser.add_argument(
            "--soft_raw",
            action="store_true",
            help="Use points2D gradients",
        )


        self.parser.add_argument("--lamda_a", type=float, default=1.,help="expansive")
        self.parser.add_argument("--lamda_s", type=float, default=1.,help="expansive")
        self.parser.add_argument("--es_beta_a", type=float, default=0.5,help="expansive")


    def _post_process(self, args):
        up_folder_name = args.up_folder_name
        if args.debug:
            up_folder_name = "debug"

        cur_time = gen_cur_time()
        random_str = gen_random_str()
        args.running_name = f"{args.tag}_{cur_time}_{random_str}"

        args.save_folder = os.path.join(
            args.save_dir, up_folder_name, args.running_name
        )
        os.makedirs(args.save_folder, exist_ok=True)

        # save to args.json
        with open(os.path.join(args.save_folder, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
            return args

    def get_args(self):
        args = self.parser.parse_args()
        args = self._post_process(args)
        return args
