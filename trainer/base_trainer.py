
from util.recorder import recorder
import torch
import torch.nn.functional as F
import os
from collections import defaultdict

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.device)
        self.data_name = os.path.splitext(os.path.basename(self.args.input_path))[0]

        recorder.dic[self.data_name] = defaultdict(dict)
        self.recorder = recorder.dic[self.data_name]

   