# RoboCOIN Object Manager

面向机器人视频数据的物体发现、跟踪、属性标注、去重和人工审核工具。项目从 RoboCOIN 数据中构建可持续更新的物体库，并为每个最终物体分配稳定 ID，例如 `cup_0`、`cup_1`。

当前版本完全在本机或数采员电脑运行，不依赖训练/推理服务器。SAM3、SAM3.1、RealESRGAN 和 CLIP 使用本地 GPU，图像理解使用 GLM 视觉 API。

## 主要功能

- 从数据集名称、任务和场景标注中提取并过滤物体词；
- 使用 RealESRGAN 增强关键帧，使用 SAM3 检测物体；
- 在中文审核页面增删关键帧、修改掩码，并用框和正负点细化；
- 使用 SAM3.1 跟踪已确认物体，自动选择质量最佳的代表图；
- 使用 GLM 标注名称和视觉属性，本地 WordNet 生成实体类别路径；
- 使用属性与 CLIP 相似度生成去重候选，通过人工树完成合并、拆分和分类；
- 构建带稳定 ID 和完整来源信息的物体库。

核心流程：

```text
视频与文本标注
    → SAM3 物体发现与人工审核
    → SAM3.1 视频跟踪与代表图选择
    → GLM 属性标注与 WordNet 分类
    → CLIP 去重候选与人工调整树
    → 稳定 ID 物体库
```

## 环境要求

- Ubuntu 22.04
- Python 3.10
- NVIDIA CUDA GPU，最低 8 GB 显存
- FFmpeg、Git
- 最低 16 GB 内存并配置交换空间
- 至少 50 GB 可用 SSD 空间，具体取决于视频数量
- 可访问 GLM API 的网络

推荐使用 12 GB 以上显存、32 GB 内存和 100 GB 以上可用空间。

## 安装

### 1. 获取代码

```bash
git clone -b sam_label https://github.com/Beihuyang/RoboCOIN-Object-Manager.git
cd RoboCOIN-Object-Manager
```

安装系统依赖：

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv ffmpeg git
```

### 2. 准备模型权重

Git 仓库不包含大模型文件。运行完整流程前，需要准备以下目录：

```text
sam3_weights/
├── sam3.pt
└── sam3.1_multiplex.pt

models/
├── clip/
│   └── ViT-L-14.pt
└── realesrgan/
    └── RealESRGAN_x2plus.pth
```

`setup.sh` 可自动下载 RealESRGAN 权重；SAM3、SAM3.1 和 CLIP 权重需要从已有部署复制或按对应项目的授权方式获取。

### 3. 配置 GLM API

```bash
cp collector.env.example collector.env
```

编辑 `collector.env`，至少填写：

```ini
ROBOCOIN_VLM_BACKEND=api
VLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
VLM_API_KEY=你的密钥
VLM_MODEL=glm-5.3-flash
```

`collector.env` 已被 Git 忽略，不要把 API 密钥提交到仓库。

### 4. 创建运行环境

```bash
chmod +x setup.sh start_collector.sh
./setup.sh
```

脚本会创建 `.venv`、安装 CUDA 版 PyTorch 和项目依赖、安装 SAM3/CLIP/WordNet，并检查 GPU 与模型文件。检查现有环境可运行：

```bash
./setup.sh --check
```

## 准备数据

把 RoboCOIN 视频和标注放在：

```text
RoboCOIN_datasets/<数据集名称>/
```

小规模测试也可以下载前 10 条数据：

```bash
source .venv/bin/activate
python download_head_videos.py --limit 10
```

如果已有其他电脑生成的审核结果，请同时复制 `objects/`。项目会使用相对路径保存新结果，并兼容迁移旧路径。

## 快速运行

启动本地审核程序：

```bash
./start_collector.sh
```

启动成功后，浏览器会自动打开。也可以手动访问：

- 掩码审核：<http://127.0.0.1:8888/review>
- 属性与去重树：<http://127.0.0.1:8888/dedup>
- 最终物体库：<http://127.0.0.1:8888/new-library>
- 文本物体 ID 替换：<http://127.0.0.1:8888/object-links>

推荐按以下顺序操作：

1. 在“掩码审核”中检查 SAM3 结果，补充关键帧并修正掩码；
2. 确认掩码后运行 SAM3.1 跟踪；跟踪直接采用审核清单中的全部关键帧和掩码，不会重新筛选名词；多个关键帧分别建立推理状态，再按物体 ID 合并轨迹；
3. 在“属性与去重树”中运行属性标注和去重；
4. 人工修正类别、物体组和垃圾项；
5. 在“最终物体库”中抽查稳定 ID、图片和来源。

批量跟踪任务在独立进程中执行；刷新或重新打开审核页面后，页面会自动恢复当前任务的进度显示。

如果需要直接运行命令行流水线：

```bash
source .venv/bin/activate
set -a && source collector.env && set +a

python stage1_track_select.py --stage discover
python stage1_track_select.py --stage track --dirty-only --continue-on-error
python run_vlm_library_pipeline.py
```

## 主要输出

```text
objects/
├── tracks/                    # 关键帧、掩码、跟踪结果和代表图
├── new_library_work/          # 属性、CLIP 特征和人工调整树
├── new_library/               # 最终物体库
└── object_id_registry.json    # 稳定 ID 注册表
```

物体 ID 一经分配不会因实例顺序变化而重编号，删除后的编号也不会复用。

## 部署到数采员电脑

先生成不包含旧审核历史和实验文件的最小完整目录，再通过 SSH 部署：

```bash
.venv/bin/python build_collector_package.py
./deploy_collector_ssh.sh username@192.168.1.120
```

部署完成后，数采员在自己的电脑进入 `RoboCOIN-Collector`，配置 `collector.env` 并运行：

```bash
./start_collector.sh
```

目标电脑随后可独立完成模型推理和人工审核，SSH 只用于首次传输与安装。详细步骤见 [快速部署指南](docs/DEPLOYMENT_GUIDE.md)，人工操作见 [数采员使用手册](docs/DATA_COLLECTOR_GUIDE.md)。

## 注意事项

- 当前不使用图像补全，`stage2_inpaint.py` 不是正式流程的一部分；
- 不要同时在多个页面修改同一条数据；
- 不要手工删除或改名 `objects/` 内的文件；
- VLM 属性和自动去重都需要人工复核；
- `.history/` 保存可恢复的审核历史，但不会进入数采员最小部署包。
