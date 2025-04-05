import os
from trainer.base_trainer import BaseTrainer
from components.tesorf.train import train_tensorf 

class NeRFTrainer(BaseTrainer):
    def __init__(self, args):
        args = self._parse_dataset(args)
        super().__init__(args)
    
    def _parse_dataset(self, args):
        '''update args.input_path'''
        data_folder = args.data_folder
        dataset = args.nerf_dataset
        name_dic = {
            "syn":"nerf_synthetic",
            "nsvf": "Synthetic_NSVF",
            "llff" : "nerf_llff_data",
            "mip360": "mip-nerf-360",
            "tt":"TanksAndTemple",
        }
        args.input_path = os.path.join(data_folder, name_dic[dataset], args.nerf_scene)
        return args

    def train(self):
        backbone = self.args.nerf_backbone
        if backbone == "tensorf":
            self._train_tensorf()
        elif backbone == "nerf":
            raise NotImplementedError
        elif backbone == "ingp":
            raise NotImplementedError

            
    def _train_tensorf(self):   
        train_tensorf(self.args, self.recorder)
        print(self.recorder)
