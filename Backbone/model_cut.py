import time
from feature_extract import *
from dataset_creater import *
from dataset.write_data_label_txt_new import *
import os
import shutil
from pathlib import Path
from timm.models import create_model
import torch.nn.utils.prune as prune
import torch
import os

# ---------------------------- 配置参数 ---------------------------- #
ckpt_path = "F:\\Backbone\\model_pkl\\vit_g_hybrid_pt_1200e_k710_ft.pth"  # 模型权重路径
save_dir = "F:\\Backbone\\model_pkl\\50"  # 保存路径
prune_rate = 0.50  # 剪枝率 10%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------- 定义 MAE 模型 ---------------------------- #
model = create_model(
    'vit_giant_patch14_224',
    img_size=224,
    pretrained=False,
    num_classes=710,
    all_frames=16,
    tubelet_size=2,
    drop_path_rate=0.3,
    use_mean_pooling=True
)

# ---------------------------- 加载预训练权重 ---------------------------- #
ckpt = torch.load(ckpt_path, map_location='cpu')
for model_key in ['model', 'module']:
    if model_key in ckpt:
        ckpt = ckpt[model_key]
        break
model.load_state_dict(ckpt)
model.to(device)

# ---------------------------- 剪枝模型（包括 Conv3D 层） ---------------------------- #
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
        prune.l1_unstructured(module, name='weight', amount=prune_rate)  # L1剪枝
        prune.remove(module, 'weight')  # 移除掩码，正式剪枝

print("模型剪枝完成（Conv2D、Conv3D、Linear 层均已剪枝）。")

# ---------------------------- 量化模型（动态量化） ---------------------------- #
# 仅对 Linear 层进行动态量化，因 Conv3D 不支持动态量化
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print("模型量化完成（8-bit 动态量化，仅针对 Linear 层）。")

# ---------------------------- 保存剪枝和量化后的模型 ---------------------------- #
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, "pruned_quantized_model.pth")
torch.save(model_quantized.state_dict(), save_path)

print(f"剪枝和量化后的模型已保存至: {save_path}")


