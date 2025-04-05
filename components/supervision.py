import torch.nn.functional as F
from components.laplacian import compute_laplacian_loss, laplacian_loss
import torchvision.transforms as transforms
import torch
from components.transform import Transform
import math


class Supervision(object):
    def __init__(self, args):
        self.args = args
        self.lambda_l = self.args.lambda_l
        self._use_laplacian_loss = bool(self.lambda_l > 0)

        self.transform = Transform(args)
    
    def set_gt(self, gt):
        self.gt = gt
        # image → 获取laplace
        self.lap_gt = laplacian_loss.laplacian(gt.unsqueeze(0).to(torch.float32))

    def compute_ratio_loss(self, pred, gt):
        error_map = F.mse_loss(pred, gt, reduction='none')
        error_map_mean = error_map.mean(1)
        ratio = self.args.ratio

        # error map %ratio
        _, indices = torch.topk(error_map_mean, int(error_map_mean.shape[0] * ratio))
        # laplace map % ratio
        # _, indices = torch.topk(self.lap_gt.flatten(), int(error_map_mean.shape[0] * ratio))

        mse = F.mse_loss(pred, gt)

        if ratio == 0:
            return mse, [mse]
         
        extra_loss = F.mse_loss(pred[indices], gt[indices])

        total_loss = mse + self.args.lamda_1 * extra_loss
        return total_loss, [mse, extra_loss]


    def _compute_loss_dev(self, pred, gt, r_pred, r_gt, epoch):
        """dev version"""
        if self.args.use_blur_sup:
            # 训不起来
            r_pred = self.guassian_blur(r_pred)
            r_gt = self.guassian_blur(r_gt)
            mse = F.mse_loss(self._encode_img(pred), self._encode_img(gt))

        else:
            mse = F.mse_loss(pred, gt)


        components = [mse]
        lamda = [1]

        if self._use_laplacian_loss:
            laplacian_loss = compute_laplacian_loss(r_pred, r_gt)
            components.append(laplacian_loss)
            lamda_laplaican = self._get_lamda_laplaican(epoch)
            lamda.append(lamda_laplaican)

        # compute total loss
        loss = 0
        for i in range(len(components)):
            loss += lamda[i] * components[i]
        return loss, components

    def guassian_blur(self, img):
        gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        blurred = gaussian_blur(img)
        return blurred

    def _encode_img(self, img):
        img = torch.clamp(img, min=0, max=255)
        img = img / 255.0
        img = self.transform.tranform(img)
        return img

    def _get_lamda_laplaican(self, epoch):
        """线性变换 目前看效果不好"""
        if self.args.lambda_l_schedular == 0:
            return self.args.lambda_l
        else:
            # linear moving
            return (
                self.args.lambda_l
                * (epoch / self.args.num_epochs)
                * self.args.lambda_l_schedular
            )

    def compute_series_loss(self, pred, gt):
        """测试spectral bias"""
        mse = F.mse_loss(pred, gt)
        loss_list = self.compute_diff_loss(pred, gt)
        loss_list = [mse] + loss_list
        _len = len(loss_list)
        _coff_list = [self.args.lamda_0,
            self.args.lamda_1,
            self.args.lamda_2] + [1 for _ in range(_len-3)]

        # 试一下数值稳定性
        _coff_list = [5, 2, 1/1.5, 1/5, 1/18, 1/65, 1/242]

        _coff_list = _coff_list[:_len]


        return loss_list, _coff_list
    

    def compute_diff_loss(self, pred, gt):
        _pred = pred.squeeze()
        _gt = gt.squeeze()

        order = self.args.order_sup
        _pred_list = self._get_derivate_list(_pred, order)
        _gt_list = self._get_derivate_list(_gt, order)

        _loss_list = []
        for (_pred, _gt) in zip(_pred_list, _gt_list):
            _loss_list.append(F.mse_loss(_pred, _gt))

        return _loss_list

    def _compute_derivative(self, x):
        # 一维差分
        return x[1:] - x[:-1]
    
    def _get_derivate_list(self, x, order):
        d_lis = []
        cur_derivate = x
        for i in range(order):
            cur_derivate = self._compute_derivative(cur_derivate)
            d_lis.append(cur_derivate)
        return d_lis
        
            

