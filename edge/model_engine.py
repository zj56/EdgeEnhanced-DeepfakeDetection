
from main import device
import math
import argparse
import kornia
import torch.nn as nn
from PIL import Image
import ipdb
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from xception import Xception
from torchvision.transforms import transforms
from srm_conv import SRMConv2d_simple
import torch.nn.init as init
import os
from torchkeras import summary
from torch import nn
import ipdb
from torch.cuda.amp import autocast, GradScaler


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def npr(image: torch.Tensor):
    n, c, w, h = image.shape
    if -1 * w % 2 != 0:
        image = image[:, :, :w % 2 * -1, :]
    if -1 * h % 2 != 0:
        image = image[:, :, :, :h % 2 * -1]

    n, c, w, h = image.shape
    if w % 2 == 1:
        image = image[:, :, :-1, :]
    if h % 2 == 1:
        image = image[:, :, :, :-1]

    factor = 0.5
    NPR = F.interpolate(image, scale_factor=factor,
                        mode='nearest', recompute_scale_factor=True)
    NPR = F.interpolate(NPR, scale_factor=1 / factor,
                        mode='nearest', recompute_scale_factor=True)
    NPR = image - NPR
    image = NPR * 2.0 / 3.0
    return image


class xception_sobel_pass_npr(nn.Module):
    def __init__(self):
        super(xception_sobel_pass_npr, self).__init__()
        self.backbone = self.build_backbone()

    def build_backbone(self):
        model_config = {"mode": "original",
                        "num_classes": 1, "inc": 3, "dropout": False}
        backbone = Xception(model_config)

        # 检查本地是否存在预训练模型文件
        model_path = "./xception-b5690688.pth"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        # 加载预训练模型
        state_dict = torch.load(model_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        return backbone

    def forward(self, x: torch.Tensor, return_feature=False):
        sobel_feature = kornia.filters.Sobel()(x)
        sobel_feature = sobel_feature + x
        npr_feature = npr(sobel_feature)
        feature = self.backbone.features(npr_feature)
        if return_feature:
            feature = nn.ReLU(inplace=True)(feature)
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            feature = feature.view(feature.size(0), -1)
            return feature
        pred = self.backbone.classifier(feature)
        return pred

