from trainer.base_trainer import BaseTrainer
from trainer.sampler import Sampler

from components.sdf import MeshSDF,generate_mesh
from types import SimpleNamespace
import torch
import numpy as np
from chamfer_distance import ChamferDistance as chamfer_dist

import mcubes
import trimesh
from tqdm import trange

from util.logger import log
from util import io
from util import open3d_utils
from util.tensorboard import writer



# todo: 基于INT的shaeINR实现的不好
## 1. resolution小
## 2. 原本输入的ply的gt有些问题 有杂质mesh 怎么解？
## 3. 采样处的没用cuda加速 导致训练起来很慢 应该先加载进cuda 再采样

class SDFTrainer(BaseTrainer,Sampler):
    def __init__(self, args):
        BaseTrainer.__init__(self, args)
        Sampler.__init__(self, args)
        self.num_samples = 50000
        self._parse_input_data()
        self.chd = chamfer_dist()
        

    def _parse_input_data(self):
        self._gen_dataset_configs()
        self._gen_input_output_configs()
        log.inst.success("start processing SDF datasets")
        self.dataset = MeshSDF(self.dataset_configs, self.input_output_configs)
        log.inst.success("finish processing SDF datasets")
    
    def _gen_dataset_configs(self):
        c = SimpleNamespace()
        c.xyz_file = self.args.input_path
        c.num_samples = self.num_samples  
        # default settings from BACON
        c.coarse_scale = 1.0e-1
        c.fine_scale = 1.0e-3
        
        c.render_resolution = 512 # 600 768

        self.dataset_configs = c
        
    def _gen_input_output_configs(self):
        c = SimpleNamespace()
        c.data_range = 0 # 貌似没用上
        c.coord_mode = 2 # 貌似没用上
        self.input_output_configs = c
    
    def _gen_mesh_from_gt_sdf(self):
        N=self.dataset.render_resolution
        sdf = self.dataset.sdf
        vertices, triangles = mcubes.marching_cubes(-sdf, 0)
        gt_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        gt_mesh.vertices = (gt_mesh.vertices / N - 0.5) + 0.5/N
        gt_o3d_mesh = open3d_utils.trimesh_to_o3d_mesh(gt_mesh)
        open3d_utils.save_mesh(gt_o3d_mesh, self._get_sub_path("gt_mesh_o3d", f"{self.data_name}_gt_{N}.ply"))
        return gt_mesh
    
    def train(self):
        self._gen_mesh_from_gt_sdf()
        num_epochs = self.args.num_epochs  # here is <iterations> 10k
        device = self.device
        model = self._get_model(in_features=3, out_features=1).to(self.device)
        optimizer = torch.optim.Adam(lr=self.args.lr, params=model.parameters())
        # align with INT
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6)

        # init_sampler
        self.model = model
        self._init_sampler()

        best_iou, best_chd, best_mesh = 0, float("inf"), None

        # use all data
        # coords, gt = self.dataset.get_all_data()
        # coords, gt = coords.to(device), gt.to(device) 
        
        # todo: 是否laplace采样？
        full_coords, full_gt = self.dataset.get_all_data()
        full_coords, full_gt = full_coords.to(device), full_gt.to(device) 

        for epoch in trange(1, num_epochs + 1):
            coords, gt, index = self.dataset.get_data()
            coords, gt, index = coords.to(device), gt.to(device),index.to(device)  #todo: tooo slow
            
            
            log.start_timer("train")
            coords, gt = self.get_sampled_coords_gt(coords, gt, epoch, index=index, full_coords=full_coords, full_gt=full_gt)
            pred = model(coords)
            self.sampler_operate_after_pred(pred, gt, epoch, recons_func=None)
            mse = self.compute_mse(pred, gt)
            loss = mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            log.pause_timer("train")

            if epoch % self.args.log_epoch == 0 or epoch == num_epochs:
                N=self.dataset.render_resolution
                # compute iou
                pred_mesh, pred_sdf = generate_mesh(model,N=N ,return_sdf=True, device=device)
                pred_occ = pred_sdf <= 0
                gt_occ = self.dataset.occu_grid
                intersection = np.sum(np.logical_and(gt_occ, pred_occ))
                union = np.sum(np.logical_or(gt_occ, pred_occ))
                iou = intersection / union

                # compute chamfer distance #todo: 这么算chamfer dis其实有问题 不应该从这个sdf里面取的
                sdf = self.dataset.sdf
                vertices, triangles = mcubes.marching_cubes(-sdf, 0)
                gt_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                gt_mesh.vertices = (gt_mesh.vertices / N - 0.5) + 0.5/N

                # compute chamfer distance
                pred_pts = torch.tensor(pred_mesh.vertices, dtype=torch.float32).unsqueeze(0).to(self.device)
                gt_pts = torch.tensor(gt_mesh.vertices,  dtype=torch.float32).unsqueeze(0).to(self.device)
                dist1, dist2, idx1, idx2 = self.chd(pred_pts,gt_pts)
                # chd_val = (torch.mean(dist1) + torch.mean(dist2)).
                chd_val1 = torch.mean(dist1).item() # 数值上这个和finer siren比较接近
                chd_val2 = torch.mean(dist2).item()
                chd_val = chd_val1 + chd_val2

                if iou > best_iou:
                    best_iou = iou
                    best_mesh = pred_mesh
                if chd_val < best_chd:
                    best_chd = chd_val
                # log
                log.inst.info(f"epoch: {epoch} → loss {loss:.12f}")
                log.inst.info(f"epoch: {epoch} → iou {iou:.12f}")
                log.inst.info(f"epoch: {epoch} → chamfer_dist1 {chd_val1:.12f}")
                log.inst.info(f"epoch: {epoch} → chamfer_dist2 {chd_val2:.12f}")
                log.inst.info(f"epoch: {epoch} → chamfer_dist {chd_val:.12f}")

                # o3d_mesh cur
                cur_o3d_mesh = open3d_utils.trimesh_to_o3d_mesh(pred_mesh)
                open3d_utils.save_mesh(cur_o3d_mesh, self._get_sub_path("log_mesh_o3d", f"{self.data_name}_{epoch}.ply"))
                
                

            # tensorboard
            writer.inst.add_scalar(
                f"{self.data_name}/train/loss", loss.detach().item(), global_step=epoch
            )
        
        log.inst.info(f"BEST iou {best_iou:.12f}")
        log.inst.info(f"BEST chamfer_dist {best_chd:.12f}")
        
        self._save_ckpt(num_epochs, model, optimizer, scheduler)
                
        
        o3d_mesh = open3d_utils.trimesh_to_o3d_mesh(best_mesh)
        open3d_utils.save_mesh(o3d_mesh,  self._get_sub_path("best_mesh_o3d", f"{self.data_name}.ply"))
    