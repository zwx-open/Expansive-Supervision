from util.logger import log
from util.tensorboard import writer
from util.plotter import plotter
from trainer.base_trainer import BaseTrainer

import numpy as np
import imageio.v2 as imageio
import torch
import os
from tqdm import trange
from functools import reduce



class SeriesTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        # self._generate_input_data()
        self._generate_phase_wave()

    def _generate_input_data(self):
        num = 3000
        signal = torch.rand(num, dtype=torch.float32)
        self.input_signal = signal
        self.T = signal.shape[0]
        plotter.plot_series(signal, self._get_cur_path("gen_signal.png"))
    
    def _generate_phase_wave(self):
        N = 200
        K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,65,70,75,80,85,90,95,100]
        PHI = [np.random.rand() for _ in K]
        t = np.arange(0, 1, 1./N)
        yt = reduce(lambda a, b: a + b, 
                    [np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, phi in zip(K, PHI)])
        
        signal = torch.from_numpy(yt).to(torch.float32)
        self.T = signal.shape[0]
        self.input_signal = signal
        plotter.plot_series(signal, self._get_cur_path("gen_signal.png"))


    def _get_data(self):
        signal = self.transform.tranform(self.input_signal)
        coords = torch.linspace(-1, 1, self.T)

        gt = signal.reshape(-1,1)
        coords = coords.reshape(-1,1)

        return coords, gt
    
    # overwrite
    def train(self):
        coords, gt = self._get_data()
    
        self.model = self._get_model(in_features=1, out_features=1).to(self.device)
        coords = coords.to(self.device)
        gt = gt.to(self.device)

        num_epochs = self.args.num_epochs

        optimizer = torch.optim.Adam(lr=self.args.lr, params=self.model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1))
        
        for epoch in trange(1, num_epochs + 1): 

            log.start_timer("train")
            
            # inference 
            pred = self.model(coords)
            mse = self.compute_mse(pred, gt)
            loss_list, lamda_list = self.supervision.compute_series_loss(pred, gt)
            loss = sum(l * lamda for l, lamda in zip(loss_list, lamda_list))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            torch.cuda.synchronize() # 确保测量时间
            log.pause_timer("train")

            # record | writer
            if epoch % self.args.log_epoch == 0:
                log.inst.info(f"epoch: {epoch} → loss {loss:.8f}")
                self.recorder[epoch]["mse"] = mse.detach().item()

                # plot 
                r_signal = self.reconstruct_signal(pred).cpu().detach()
                plotter.plot_mutil_series([self.input_signal, r_signal], self._get_sub_path("compare", f"compare_{epoch}.png"))
                
            writer.inst.add_scalar("train/total_loss", loss.detach().item(), global_step=epoch)
            writer.inst.add_scalar("train/mse", mse.detach().item(), global_step=epoch)

            for i,loss in enumerate(loss_list):
                writer.inst.add_scalar(f"train/loss_{i}", loss.detach().item(), global_step=epoch)



        # save ckpt
        # self._save_ckpt(num_epochs, self.model, optimizer, scheduler)
    
    def reconstruct_signal(self, data):
        return self.transform.inverse(data)