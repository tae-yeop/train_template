import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

class PerceptualLoss()

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, p_hat, r_tgt, m):
        return self.mse(m*p_hat, m*r_tgt)

# GAN Losses
def g_nosaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

def g_hinge(d_logit_fake):
    

class DenoiserLoss(nn.Module):
    def __init__(self, loss_args, device):
       super().__init__()
       self.loss_args = loss_args
       vgg_loss_dict = self.loss_args['multi_vgg_loss']
       # dict에 담으면 장치에 제대로 가지 않음
       self.loss_dict = {"smooth_l1_loss": PerPixelLoss().to(device),
                         "proxy_loss":proxy_loss,
                         "direction_loss":DirectionalLoss().to(device),
                         "dist_loss":DistributionLoss().to(device),
                         "perceptual_loss": multi_VGGPerceptualLoss(vgg_loss_dict['l1_lambda'], vgg_loss_dict['vgg_lambda']).to(device)}
    def forward(self, clean_hat1, clean_hat2, clean_hat3, clean_img, clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3):
        total_loss = 0
        # out1, out2, out3, gt1, feature_layers=[2]):
        total_loss += self.loss_dict["perceptual_loss"](clean_hat1, clean_hat2, clean_hat3, clean_img)
        total_loss += self.loss_dict["perceptual_loss"](clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3, clean_img)
        # for loss_name, loss_info in self.config["losses"].items():
        #     if loss_info["active"]:
        #         loss_fn = self.loss_functions[loss_name]
        #         loss_weight = loss_info["lambda"]
        #         loss_value = loss_fn(predictions, targets)
        #         total_loss += loss_weight * loss_value
        return total_loss



# 다중 모델 사용하는걸 고려
class ModelLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss_dict = {"": }
        self.loss
    def forward(self, out_dict, tgt_dict):
        total_loss = 0
        v
        params = inspect.signature(Denoiser).parameters
    print(params)
    model_params = {k: v for k, v in args.denoiser_model.items() if k != 'name'}