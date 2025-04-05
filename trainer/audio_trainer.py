from trainer.base_trainer import BaseTrainer
from trainer.sampler import Sampler
import torchaudio
import torch
from util.logger import log
import numpy as np
import os
from tqdm import trange

import pesq
from pystoi import stoi
import soundfile as sf


import matplotlib.pyplot as plt

class AudioTrainer(BaseTrainer,Sampler):
    def __init__(self, args):
        BaseTrainer.__init__(self, args)
        Sampler.__init__(self, args)
        self.sample_rate = 16000
        self.crop_sec = 5 
        self.data_name = os.path.basename(self.args.input_path).split(".")[0]
        self._parse_input_data()
    
    def _parse_input_data(self):
        audio_path = self.args.input_path
        waveform, orginal_sr = torchaudio.load(audio_path)
        if orginal_sr != self.sample_rate:
            log.inst.warning(f"inconsistent sample rate: original_sr is {orginal_sr}")
            _trans = torchaudio.transforms.Resample(orginal_sr, self.sample_rate)
            waveform = _trans(waveform)
            
        crop_samples = self.sample_rate * self.crop_sec
        self.input_audio = waveform.squeeze()
        final_len = min(crop_samples,len(self.input_audio))
        
        self.input_audio = self.input_audio[:final_len]
        self.T = final_len

    def _get_data(self, mark="default", use_normalization=True):
        # parse input audio
        audio = self.input_audio
        audio = self.transform.tranform(audio)

        gt = audio
        coords = torch.linspace(-1, 1, self.T)

        gt = gt.reshape(-1,1)
        coords = coords.reshape(-1,1)

        ######################## test 
        # r_audio = self.reconstruct_audio(gt)
        # mse = self.compute_mse(r_audio, self.input_audio)
        # snr = self.compute_snr(r_audio, self.input_audio)
        # pesq = self.compute_pesq(r_audio, self.input_audio)
        # stoi = self.compute_stoi(r_audio, self.input_audio)
        # print("Testing metrics and pre-processing .... ")
        # print('stoi: ', stoi)
        # print('pesq: ', pesq)
        # print('mse: ', mse)
        # print('snr: ', snr)
        return coords, gt
    
    def reconstruct_audio(self, pred):
        pred = pred.squeeze()
        r_audio = self.transform.inverse(pred)
        return r_audio
    
    def compute_snr(self, pred, gt):    
        noise = gt - pred
        gt_power = torch.mean(gt**2)
        noise_power = torch.mean(noise**2)
        snr = 10 * torch.log10(gt_power / noise_power)
        return snr
    
    def compute_si_snr(self, pred, gt):
        gt = gt - gt.mean()
        pred = pred - pred.mean()
        return self.compute_snr(pred, gt)
    
    def compute_stoi(self, pred, gt):
        return stoi(gt, pred, self.sample_rate, extended=False)
    
    def compute_pesq(self, pred, gt):
        pesq_score = pesq.pesq(self.sample_rate, gt.numpy(), pred.numpy(), mode='wb')
        return pesq_score
    

    def train(self):
        num_epochs = self.args.num_epochs
        coords, gt = self._get_data()
        model = self._get_model(in_features=1, out_features=1).to(self.device)

        full_coords = coords.to(self.device)
        full_gt = gt.to(self.device)

        optimizer = torch.optim.Adam(lr=self.args.lr, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1)
        )

        # init_sampler
        self.model = model
        self._init_sampler()

        for epoch in trange(1, num_epochs + 1):
            
            log.start_timer("train")
            coords, gt = self.get_sampled_coords_gt(full_coords, full_gt, epoch)
            pred = model(coords)
            
            self.sampler_operate_after_pred(pred, gt, epoch, recons_func=None)
            
            loss = self.compute_mse(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            log.pause_timer("train")

            if epoch % self.args.log_epoch == 0:
                
                # inference by all 
                with torch.no_grad():
                    full_pred =  model(full_coords)
                
                r_audio = self.reconstruct_audio(full_pred).detach().cpu()
                mse = self.compute_mse(r_audio, self.input_audio).item()
                snr = self.compute_snr(r_audio, self.input_audio).item()
                si_snr = self.compute_si_snr(r_audio, self.input_audio).item()
                pesq = self.compute_pesq(r_audio, self.input_audio)
                stoi = self.compute_stoi(r_audio, self.input_audio)

                log.inst.info(f"epoch: {epoch} → mse {mse:.8f}")
                log.inst.info(f"epoch: {epoch} → snr {snr:.4f}")
                log.inst.info(f"epoch: {epoch} → si_snr {si_snr:.4f}")
                log.inst.info(f"epoch: {epoch} → pesq {pesq:.4f}")
                log.inst.info(f"epoch: {epoch} → stoi {stoi:.4f}")

                self.recorder[epoch]["mse"] = mse
                self.recorder[epoch]["snr"] = snr
                self.recorder[epoch]["sisnr"] = si_snr
                self.recorder[epoch]["stoi"] = stoi
                self.recorder[epoch]["pesq"] = pesq
                error = full_gt - full_pred
                error = error.cpu()
                torch.save(error, self._get_sub_path("error_tensor_pts", f"error_{self.data_name}_{epoch}.pt"))
                self.plot_error_signal(error.cpu(),epoch)
                sf.write(self._get_sub_path("full_pred_audio",f"{epoch}_{self.data_name}.wav"), full_pred.cpu(), self.sample_rate)

        # with torch.no_grad():
        #     pred = model(full_coords)
        #     error = full_gt - pred
        #     sf.write(self._get_cur_path(f"final_pred_{self.data_name}.wav"), pred.cpu(), self.sample_rate)
        #     self.plot_error_signal(error.cpu(), epoch)

        self._save_ckpt(epoch, model, optimizer, scheduler)




    def plot_error_signal(self, error_signal, epoch):
        sr = self.sample_rate
        out_path = self._get_sub_path("plot_error", f"error_{self.data_name}_{epoch}.png")
        time_axis = np.linspace(0, len(error_signal) / sr, num=len(error_signal))
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, error_signal, label="Error Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Error Signal Waveform")
        plt.legend()
        plt.grid()
        plt.savefig(out_path)
