# RoboCOIN Object Manager 快速部署指南

本指南用于把程序、模型、数据集和全部人工结果迁移到另一台 Ubuntu 机器。

部署完成后，新机器会独立保存数据、运行模型和显示网页。旧机器只在第一次复制文件时使用；复制完成后可以关机，不需要用旧机器连接、访问或控制新机器。

## 1. 推荐配置

| 项目 | 最低 | 推荐 |
|---|---|---|
| 系统 | Ubuntu 22.04，64位 | Ubuntu 22.04，64位，与当前机器一致 |
| Python | 3.10 | 3.10 |
| GPU | NVIDIA CUDA，8 GB显存 | 16 GB及以上显存 |
| 内存 | 16 GB，并有16 GB交换空间 | 32 GB及以上 |
| 磁盘 | SSD，部署后剩余50 GB | NVMe SSD，剩余100 GB以上 |

当前需要迁移的数据约34 GB，其中模型约13 GB、原始视频约18 GB、人工结果约2.6 GB。迁移前确认目标机器有足够空间。

## 2. 推荐：部署数采员独立完整版

两台机器先连接同一个局域网，推荐使用网线。先确认旧机器的页面没有正在运行 SAM3、跟踪、VLM 或 CLIP 任务，同步期间不要继续标注。

### 新机器只需准备一次

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
hostname -I
```

记下 `hostname -I` 显示的局域网 IP，例如 `192.168.1.120`。这是一键脚本无法代替的唯一次新机器准备，因为机器首先必须开启 SSH，旧机器才能连接它。

### 旧机器执行一条命令

在项目根目录先生成最小完整部署目录，再把用户名和 IP 换成新机器的实际信息：

```bash
cd /home/hy/baai/RoboCOIN-Object-Manager
.venv/bin/python build_collector_package.py
chmod +x deploy_collector_ssh.sh
./deploy_collector_ssh.sh username@192.168.1.120
```

脚本运行时可能询问新机器的登录密码和 `sudo` 密码，正常输入即可。密码不会被保存。

默认部署到新机器用户的：

```text
/home/用户名/RoboCOIN-Collector
```

如果要放到用户主目录下的其他位置，可以传入第二个参数：

```bash
./deploy_collector_ssh.sh username@192.168.1.120 projects/RoboCOIN-Collector
```

一键脚本会自动：

1. 测试两台机器的 SSH 连接；
2. 在新机器安装 Python 3.10、FFmpeg、Git 和 rsync；
3. 同步数采所需的代码、模型、数据集、缓存和当前人工结果；审核历史 `.history/` 与旧跟踪归档 `_tracker_archive/` 不进入部署包；
4. 不复制旧机器的 `.venv`，而是在新机器重新创建；
5. 执行路径迁移、CUDA 和模型检查。

网络中断时，重新执行同一条命令即可续传。脚本不使用 `rsync --delete`，不会删除新机器目录中已有的文件。

## 3. 部署完成后在新机器独立运行

部署后，在新机器复制配置模板并由技术人员填写 GLM API 密钥：

```bash
cd RoboCOIN-Collector
cp collector.env.example collector.env
# 编辑 collector.env，填写真实 VLM_API_KEY
./start_collector.sh
```

保持终端运行，在**新机器自己的浏览器**打开：

```text
http://127.0.0.1:8888/review
```

启动脚本会先检查 CUDA、依赖和模型，随后启动页面并自动打开浏览器。至此程序完全依靠新机器运行；旧机器不需要开机或端口转发。关闭启动窗口会停止页面。

## 4. 必须迁移的内容

必须复制：

| 目录 | 内容 |
|---|---|
| `sam3/` | 项目使用并修改过的SAM3代码 |
| `sam3_weights/` | SAM3与SAM3.1权重 |
| `models/` | CLIP和RealESRGAN权重；GLM通过API调用，不携带Qwen权重 |
| `RoboCOIN_datasets/` | 原始视频 |
| `objects/` | 掩码、跟踪结果、属性缓存、人工调整树和物体库 |
| `reports/` | 效果评估报告和对比图 |
| `docs/`及项目脚本 | 程序和操作文档 |

不要复制 `.venv/`。虚拟环境包含旧机器路径和 CUDA 相关二进制，应在新机器重新创建。一键脚本已自动处理这一点。

## 5. 手动部署（仅用于排错）

如果一键脚本失败，可按下面步骤分段执行，用于定位是连接、传输还是环境安装问题。

### 5.1 旧机器手动同步

```bash
rsync -a --partial --info=progress2 \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.log' \
  /home/hy/baai/RoboCOIN-Collector/ \
  username@192.168.1.120:RoboCOIN-Collector/
```

### 5.2 新机器安装系统依赖

先检查NVIDIA驱动：

```bash
nvidia-smi
```

安装基础工具：

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv ffmpeg git
```

进入项目目录：

```bash
cd RoboCOIN-Collector
```

### 5.3 创建项目环境

```bash
chmod +x setup.sh
PYTHON_BIN=python3.10 ./setup.sh
```

脚本会：

- 创建新的 `.venv`；
- 安装项目依赖和SAM3源码包；
- 安装OpenAI CLIP和WordNet；
- 检查或下载RealESRGAN_x2plus；
- 自动迁移旧绝对路径；
- 检查CUDA和必要模型文件。

脚本默认安装CUDA 12.8版PyTorch。如果目标机器需要其他版本，可指定对应官方源，例如：

```bash
PYTHON_BIN=python3.10 \
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
./setup.sh
```

### 5.4 验证部署

```bash
./setup.sh --check
```

正常结果应显示：

- Python和PyTorch版本；
- `CUDA: True`；
- 正确的GPU名称；
- 没有缺失的SAM3、SAM3.1、CLIP或RealESRGAN文件。

再检查关键目录：

```bash
du -sh sam3_weights models RoboCOIN_datasets objects reports
```

### 5.5 启动页面

首次验证建议在前台启动：

```bash
./start_collector.sh
```

在新机器本地打开：

```text
http://127.0.0.1:8888/review
```

无需从旧机器访问新机器，也无需配置端口转发。

## 6. 迁移完成检查

依次确认：

1. `/review` 的审核数据数量与旧机器一致（当前为809条）；
2. 关键帧、人工框、正负点和掩码能够正常显示；
3. “等待跟踪”和“跟踪已更新”状态正确；
4. `/dedup` 能看到原来的人工分簇、类别和垃圾桶状态；
5. `/new-library` 能打开物体图片和来源；
6. `reports/` 中的报告和对比图可以打开。

目录位置改变不会导致新缓存失效。项目内的新JSON使用相对路径，读取程序也会自动重新定位旧结果。

## 7. 常见问题

### SSH 连接失败

确认两台机器在同一局域网，新机器已执行 `sudo systemctl enable --now ssh`，并检查用户名和 `hostname -I` 显示的 IP。

```bash
ssh username@192.168.1.120
```

这条命令可以登录后，再重新运行一键脚本。

### `CUDA: False`

检查NVIDIA驱动和PyTorch CUDA版本。不要继续运行SAM3或VLM任务。

### 提示缺少模型

重新同步 `sam3_weights/` 和 `models/`。`git clone` 不会下载这些目录。

### 页面能打开但图片不存在

确认 `RoboCOIN_datasets/` 和 `objects/` 已完整同步，再运行：

```bash
.venv/bin/python migrate_paths.py
.venv/bin/python migrate_paths.py --apply
```

### 端口8888被占用

```bash
.venv/bin/python viewer.py --port 8889
```

随后在新机器本地浏览器访问 `http://127.0.0.1:8889/review`。

## 8. 无局域网时使用移动硬盘

把项目目录复制到移动硬盘，再复制到新机器。同样不要复制 `.venv/`。复制完后，在新机器运行：

```bash
cd RoboCOIN-Collector
chmod +x setup.sh
PYTHON_BIN=python3.10 ./setup.sh
./setup.sh --check
```
