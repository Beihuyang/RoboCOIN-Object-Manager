# A800 服务器推理模式

服务器模式只负责 SAM3、SAM3.1、Qwen3-VL 和 CLIP 的 GPU 推理，不调用
`stage2_inpaint.py`。界面、下载、候选对计算、物体库构建和类别树生成仍在本机完成。

## 1. 上传到持久化目录

在本机执行（主机别名按自己的 SSH config 修改）：

```bash
rsync -av --progress --exclude .git --exclude __pycache__ \
  ./ BAAI-job1:/share/project/robocoin/RoboCOIN-Object-Manager/
```

模型、待推理帧和服务器输出也必须位于 `/share/project`，否则 job 重启会丢失。

## 2. 选择分配的第 7 张物理卡

在 job 内执行：

```bash
cd /share/project/robocoin/RoboCOIN-Object-Manager
export CUDA_VISIBLE_DEVICES=6
```

进程此时只能看到一张卡，因此程序参数使用逻辑编号 `--gpu 0`。入口会拒绝未设置
显卡或同时暴露多张显卡的情况，避免占用其他人的卡。

## 3. 运行 GPU 推理

```bash
# SAM3：官方多 query 批量单帧检测（每批 16 个独立名词）
python3 run_server_inference.py discover --limit 100

# SAM3：人工框/点复核后的重新分割
python3 run_server_inference.py refine --dirty-only

# SAM3.1：掩码视频跟踪，视频帧和状态保留在 80GB 显存
python3 run_server_inference.py track --dirty-only --continue-on-error

# Qwen3-VL：默认 batch size 8
python3 run_server_inference.py attributes

# CLIP：默认 batch size 64，只输出特征，不在服务器做两两匹配和建库
python3 run_server_inference.py clip
```

显存不足时可单独降低批量大小，例如：

```bash
python3 run_server_inference.py attributes --batch-size 4
python3 run_server_inference.py clip --batch-size 32
```

SAM3 的批量 query 数由 `hardware_profiles.py` 中的
`sam3_prompt_batch_size` 控制；单个 query 的结果与逐词调用保持独立。

## 4. 下载输出并在本机继续

```bash
rsync -av --progress \
  BAAI-job1:/share/project/robocoin/RoboCOIN-Object-Manager/objects/ ./objects/
```

CLIP 特征下载后，可在本机继续候选匹配与建库。完整的本机流水线仍可运行：

```bash
python3 stage4_dedup.py --hardware-profile local --reuse-embeddings
python3 dedup_tree.py --regenerate

# 或让本机重新执行包含 CLIP 推理在内的完整流水线
python3 run_vlm_library_pipeline.py --hardware-profile local
```

不经过服务器入口时，也可以为单个命令显式指定
`--hardware-profile a800`，或设置 `ROBOCOIN_HARDWARE_PROFILE=a800`。
