<p align="center">
  <img src="assets/pics/logo.png"  height=100>
</p>

<div align="center">
<h2><font color="red"> HunyuanPortrait </font></center> <br> <center>Implicit Condition Control for Enhanced Portrait Animation</h2>

<a href='https://arxiv.org/abs/2503.18860'><img src='https://img.shields.io/badge/ArXiv-2503.18860-red'></a> 
<a href='https://kkakkkka.github.io/HunyuanPortrait/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://huggingface.co/tencent/HunyuanPortrait'><img src="https://img.shields.io/static/v1?label=HuggingFace&message=HunyuanPortrait&color=yellow"></a>
</div>

```bash
git clone https://huggingface.co/datasets/svjack/Xiang_Float_After_Tomorrow_Head_SPLITED_Captioned
git clone https://huggingface.co/datasets/svjack/Xiang_Card_After_Tomorrow_Adjust_Static_LatentSync_Videos

#!/bin/bash

# 定义目录和图片路径
#VIDEO_DIR="Xiang_Float_After_Tomorrow_Head_SPLITED_Captioned"
VIDEO_DIR="Xiang_Card_After_Tomorrow_Adjust_Static_LatentSync_Videos"
IMAGE_PATH="wanye.jpeg"

# 使用find命令安全地处理包含空格的文件名
while IFS= read -r -d '' video_path; do
    echo "Processing video: $video_path"

    # 执行推理命令
    python inference.py \
        --config config/hunyuan-portrait.yaml \
        --video_path "$video_path" \
        --image_path "$IMAGE_PATH"

    echo "Finished processing: $video_path"
    echo "----------------------------------"
done < <(find "$VIDEO_DIR" -name "*.mp4" -print0 | sort -z)

echo "All videos processed!"

import os
import re
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.fx.all import speedx  # 直接导入函数

# 输入和输出路径
mp3_dir = "After_Tomorrow_SPLITED"
mp4_dir = "tmp"
output_dir = "Kigurumi_HunyuanPortrait_After_Tomorrow_SPLITED_Captioned"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def extract_number(filename):
    """从文件名中提取数字前缀（如 '0001_明天过后.mp3' -> '0001'）"""
    match = re.findall(r'(\d+)_明天过后', filename)
    return match[0] if match else None

# 构建数字前缀到文件的映射
mp3_files = {}
for f in os.listdir(mp3_dir):
    if f.endswith('.mp3'):
        num = extract_number(f)
        if num:
            mp3_files[num] = os.path.join(mp3_dir, f)

mp4_files = {}
for f in os.listdir(mp4_dir):
    if f.endswith('.mp4'):
        num = extract_number(f)
        if num:
            mp4_files[num] = os.path.join(mp4_dir, f)

# 获取所有共同的前缀数字
common_numbers = set(mp3_files.keys()) & set(mp4_files.keys())

if not common_numbers:
    print("错误: 没有找到可以匹配的MP3和MP4文件")
    exit()

# 按数字顺序处理
for num in sorted(common_numbers):
    mp3_path = mp3_files[num]
    mp4_path = mp4_files[num]
    output_path = os.path.join(output_dir, os.path.basename(mp4_path))
    
    print(f"正在处理: {num} (MP4: {os.path.basename(mp4_path)}, MP3: {os.path.basename(mp3_path)})")
    
    # 加载音频和视频
    audio = AudioFileClip(mp3_path)
    video = VideoFileClip(mp4_path)
    
    # 计算需要的速度因子
    original_duration = video.duration
    target_duration = audio.duration
    speed_factor = original_duration / target_duration
    
    # 调整视频速度
    if speed_factor != 1.0:
        print(f"  调整视频速度: {speed_factor:.2f}x")
        video = video.fx(speedx, speed_factor).set_duration(audio.duration) 
    
    # 设置音频
    video = video.set_audio(audio)
    
    # 写入输出文件
    video.write_videofile(
    output_path,
    codec='libx264',
    audio_codec='aac',      # 必须取消注释
    temp_audiofile='temp-audio.m4a',
    remove_temp=True,
    threads=4
    )
    
    # 关闭剪辑以释放资源
    audio.close()
    video.close()

print(f"处理完成！共处理了 {len(common_numbers)} 个文件")

```

## 🧩 Community Contributions
If you develop/use HunyuanPortrait in your projects, welcome to let us know/sumbit a PR! 💖

## 📜 Requirements
* An NVIDIA 3090 GPU with CUDA support is required. 
  * The model is tested on a single 24G GPU.
* Tested operating system: Linux

## 📥 Installation

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanPortrait
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

## 🛠️ Download

All models are stored in `pretrained_weights` by default:
```bash
pip3 install "huggingface_hub[cli]"
cd pretrained_weights
huggingface-cli download --resume-download stabilityai/stable-video-diffusion-img2vid-xt --local-dir . --include "*.json"
wget -c https://huggingface.co/LeonJoe13/Sonic/resolve/main/yoloface_v5m.pt
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors -P vae
wget -c https://huggingface.co/FoivosPar/Arc2Face/resolve/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx
huggingface-cli download --resume-download tencent/HunyuanPortrait --local-dir hyportrait
```

And the file structure is as follows:
```bash
.
├── arcface.onnx
├── hyportrait
│   ├── dino.pth
│   ├── expression.pth
│   ├── headpose.pth
│   ├── image_proj.pth
│   ├── motion_proj.pth
│   ├── pose_guider.pth
│   └── unet.pth
├── scheduler
│   └── scheduler_config.json
├── unet
│   └── config.json
├── vae
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
└── yoloface_v5m.pt
```

## ▶️ Run

🔥 Live your portrait by executing `bash demo.sh`

```bash
video_path="your_video.mp4"
image_path="your_image.png"

python inference.py \
    --config config/hunyuan-portrait.yaml \
    --video_path $video_path \
    --image_path $image_path
```

## 🏗️ Framework 
<img src="assets/pics/pipeline.png">

## ⏳ TL;DR:
HunyuanPortrait is a diffusion-based framework for generating lifelike, temporally consistent portrait animations by decoupling identity and motion using pre-trained encoders. It encodes driving video expressions/poses into implicit control signals, injects them via attention-based adapters into a stabilized diffusion backbone, enabling detailed and style-flexible animation from a single reference image. The method outperforms existing approaches in controllability and coherence.

# 🖼 Gallery

Some results of portrait animation using HunyuanPortrait.

More results can be found on our [Project page](https://kkakkkka.github.io/HunyuanPortrait/).

## 📂 Cases

<table>
<tr>
<td width="25%">
  
https://github.com/user-attachments/assets/b234ab88-efd2-44dd-ae12-a160bdeab57e

</td>
<td width="25%">

https://github.com/user-attachments/assets/93631379-f3a1-4f5d-acd4-623a6287c39f

</td>
<td width="25%">

https://github.com/user-attachments/assets/95142e1c-b10f-4b88-9295-12df5090cc54

</td>
<td width="25%">

https://github.com/user-attachments/assets/bea095c7-9668-4cfd-a22d-36bf3689cd8a

</td>
</tr>
</table>

## 🎤 Portrait Singing

https://github.com/user-attachments/assets/4b963f42-48b2-4190-8d8f-bbbe38f97ac6

## 🎬 Portrait Acting

https://github.com/user-attachments/assets/48c8c412-7ff9-48e3-ac02-48d4c5a0633a

## 🤪 Portrait Making Face

https://github.com/user-attachments/assets/bdd4c1db-ed90-4a24-a3c6-3ea0b436c227

## 💖 Acknowledgements

The code is based on [SVD](https://github.com/Stability-AI/generative-models), [DiNOv2](https://github.com/facebookresearch/dinov2), [Arc2Face](https://github.com/foivospar/Arc2Face), [YoloFace](https://github.com/deepcam-cn/yolov5-face). We thank the authors for their open-sourced code and encourage users to cite their works when applicable.
Stable Video Diffusion is licensed under the Stable Video Diffusion Research License, Copyright (c) Stability AI Ltd. All Rights Reserved.
This codebase is intended solely for academic purposes.

# 🔗 Citation 
If you think this project is helpful, please feel free to leave a star⭐️⭐️⭐️ and cite our paper:
```bibtex
@article{xu2025hunyuanportrait,
  title={HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation},
  author={Xu, Zunnan and Yu, Zhentao and Zhou, Zixiang and Zhou, Jun and Jin, Xiaoyu and Hong, Fa-Ting and Ji, Xiaozhong and Zhu, Junwei and Cai, Chengfei and Tang, Shiyu and Lin, Qin and Li, Xiu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2503.18860},
  year={2025}
}
``` 
