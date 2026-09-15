# RoboCOIN Object Manager

使用 A800 job 只执行模型推理时，参见 [SERVER_MODE.md](SERVER_MODE.md)。服务器
入口不会运行图像补全，并使用单卡 80GB 高显存配置。

从 RoboCOIN 视频构建物体库的实验性流水线。项目目标是：默认在视频首帧检测物体，必要时由人工追加多张关键帧补充后续出现的物体，随后跟踪这些物体、选择代表帧、提取属性、生成去重候选，并通过人工界面审核结果。

面向实际标注与审核人员的操作说明见：[数采员使用手册](docs/DATA_COLLECTOR_GUIDE.md)。迁移到其他机器时参照：[快速部署指南](docs/DEPLOYMENT_GUIDE.md)。

## 当前状态

| 阶段 | 文件 | 状态 |
|---|---|---|
| 数据下载 | `download_head_videos.py` | 可用；本地已有 1 个数据集、2 个头部相机视频 |
| 多关键帧物体发现 | `stage1_track_select.py` | SAM 3 默认发现源视频第 0 帧，支持人工追加关键帧补充物体 |
| 视频跟踪与选帧 | `stage1_track_select.py` | SAM 3.1 multiplex；支持多关键帧掩码注入和代表图选帧 |
| 完整轮廓/完整 RGB | `stage2_inpaint.py` | 原型不可直接使用，见“Stage 2 限制” |
| 属性提取 | `stage3_attribute.py` | 对质量最高图标注类别、颜色、材质、形状、纹理 |
| 属性与视觉聚簇 | `stage4_dedup.py`、`dedup_tree.py` | 属性树预分组，ViT-L/14 complete-link 聚簇 |
| 人工审核与物体库浏览 | `viewer.py` | 支持拖拽调整簇、跨节点移动、垃圾桶软删除及属性修改 |
| 离线人工审核包 | `export_offline_review.py` / `import_offline_review.py` | 导出选中视角视频、关键帧、掩码、中文名词和本地 SAM3 框点细化环境，不携带完整数据集与跟踪模型；数采员回传后由导入工具做带基线校验的增量合并 |

## 硬件与环境

- Ubuntu，Python 3.10.12
- NVIDIA GeForce RTX 4070 Laptop GPU，8 GB 显存
- PyTorch 2.10.0 + CUDA 12.8
- 项目虚拟环境：`.venv`

完整流水线的可运行下限是 NVIDIA CUDA GPU 8 GB 显存、16 GB 内存加交换空间、8 核 CPU 和任务开始前至少 50 GB 可用 SSD 空间；该配置必须串行运行模型任务。推荐 12 GB 及以上显存、32 GB 内存、12 核及以上 CPU 和 100 GB 以上可用 NVMe 空间。项目模型与虚拟环境约占 14 GB，数据、抽帧、2K 图和缓存需另行预留空间。

新部署使用隔离的 `.venv`，并从指定的 PyTorch CUDA 软件源安装独立运行环境，不再继承宿主 Python 包。`setup.sh --check` 会严格检查 Python 3.10、CUDA、必要依赖和模型文件；任一关键项缺失都会返回失败，避免部署显示成功但模型不能运行。

已下载的公开权重：

```text
models/
├── Qwen3-VL-2B-Instruct/      # 约 4.0 GiB
├── realesrgan/
│   └── RealESRGAN_x2plus.pth # SAM3 前置 2K 超分权重，约 64 MiB
├── sam2-hiera-large/          # 约 898 MB
└── clip/ViT-B-16.pt           # 约 335 MiB
sam3_weights/
├── sam3.pt                    # SAM 3：首帧发现和人工框细化
└── sam3.1_multiplex.pt        # SAM 3.1：multiplex 掩码跟踪
```

`models/` 已加入 `.gitignore`，不会误提交大文件。

进入项目并激活环境：

```bash
cd /home/hy/baai/RoboCOIN-Object-Manager
source .venv/bin/activate
```

验证环境：

```bash
python - <<'PY'
import cv2, numpy, torch
from sam3.model_builder import build_sam3_video_predictor
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("SAM 3 code import: OK")
PY
```

退出环境：

```bash
deactivate
```

### 换电脑时的一键安装

数采员电脑使用 GLM API 时，推荐先生成精简但功能完整的独立目录，再通过 SSH 部署：

```bash
.venv/bin/python build_collector_package.py
./deploy_collector_ssh.sh username@192.168.1.120
```

生成目录为同级的 `RoboCOIN-Collector/`，包含原始视频、当前审核结果、SAM3/SAM3.1、CLIP、RealESRGAN、GLM API 后端和全部人工页面；不包含 Qwen 本地权重、SAM2、报告实验、服务器/离线回传工具、审核历史目录 `.history/` 和旧跟踪归档 `_tracker_archive/`。历史和归档仍保留在源项目中，只是不传给数采电脑。大文件在本机暂存目录中使用硬链接节省空间，通过 SSH 传到数采电脑后是独立文件。

目标电脑复制 `collector.env.example` 为 `collector.env` 并填写 GLM API 密钥，日常只需运行：

```bash
./start_collector.sh
```

如果两台机器在同一局域网，可以在旧机器的项目根目录直接执行：

```bash
chmod +x deploy_remote.sh
./deploy_remote.sh username@192.168.1.120
```

`deploy_remote.sh` 通过 SSH 和 rsync 同步代码、模型、数据、缓存和人工结果，然后在新机器调用 `setup.sh` 创建环境并检查部署。新机器需要先开启 SSH，完整说明见 [快速部署指南](docs/DEPLOYMENT_GUIDE.md)。

SSH 只用于首次传输和安装。部署结束后，在新机器本机运行 `python viewer.py`，并用新机器自己的浏览器访问 `http://127.0.0.1:8888/review`；旧机器无需继续开机、连接或访问新机器。

如果项目已经通过移动硬盘或其他方式复制到新机器，只需在项目根目录运行：

```bash
chmod +x setup.sh
./setup.sh
```

脚本会创建 `.venv`、安装项目依赖、SAM3 源码包、OpenAI CLIP 和
WordNet，并自动下载、校验 RealESRGAN_x2plus 公开权重。默认使用 CUDA 12.8 的 PyTorch wheel；其他
CUDA 环境可在执行前指定，例如：

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 ./setup.sh
```

只检查现有环境、不安装任何内容：

```bash
./setup.sh --check
```

除 RealESRGAN_x2plus 外，其他模型、数据和生成结果不由安装脚本自动下载。要完整保留当前工作，需
同时复制 `models/`、`sam3_weights/`、`RoboCOIN_datasets/` 和 `objects/`。

### 可迁移的结果路径

新生成的 JSON 元数据把项目内路径保存为相对于项目根目录的路径，因此整个项目
移动到另一目录或另一台电脑后仍可读取。读取代码也兼容旧的绝对路径，并会根据
`objects/`、`RoboCOIN_datasets/` 等目录自动重新定位。

旧结果可用下面的命令转换；先预览，再真正写入：

```bash
python migrate_paths.py
python migrate_paths.py --apply
```

## 从零重建虚拟环境

Ubuntu 当前缺少 `python3.10-venv`，因此使用 `virtualenv`。机器已有可用的 CUDA 版 PyTorch，所以使用 `--system-site-packages` 复用它：

```bash
python3 -m pip install --user --upgrade virtualenv
python3 -m virtualenv --clear --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-project.txt
python -m pip install -e ./sam3
python -m pip install git+https://github.com/openai/CLIP.git
python -m nltk.downloader wordnet
```

如果换到没有预装 PyTorch 的机器，需要先按对应 CUDA 版本安装 PyTorch 和 TorchVision。

重新下载公开权重：

```bash
python -m modelscope.cli.cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir models/Qwen3-VL-2B-Instruct --max-workers 4

python -m modelscope.cli.cli download AI-ModelScope/sam2-hiera-large \
  config.json model.safetensors preprocessor_config.json \
  processor_config.json video_preprocessor_config.json \
  --local-dir models/sam2-hiera-large

python - <<'PY'
from pathlib import Path
import clip
root = Path("models/clip")
root.mkdir(parents=True, exist_ok=True)
clip.load("ViT-B/16", device="cpu", download_root=str(root))
PY
```

## 数据下载

如果目标是把任务和标注中的物体名词链接到物体库 ID，只下载可能包含物体自然语言的文件：

```bash
python download_object_text.py
```

该命令只下载 `meta/episodes.jsonl`、`meta/tasks.jsonl`、`annotations/scene_annotations.jsonl`、`annotations/subtask_annotations.jsonl` 和 `annotations/subtasks.jsonl`。`data/*.parquet` 没有文本列，主要是关节状态、动作、时间戳和标注索引，因此不会下载；EEF、夹爪、相机参数和统计文件也不在此任务的下载范围内。

物体库完成永久 `category_N` ID 建库后，生成不覆盖原始 JSONL 的名词替换结果：

```bash
python object_text_linker.py --generate
python viewer.py
```

结果写入 `RoboCOIN_object_linked/`，目录结构与原数据一致；原文本同时保存在 `source_task`、`source_tasks`、`source_scene` 或 `source_subtask`。只有“当前数据集内唯一候选”会自动替换成 `[cup_0]`；多候选保留原词并进入 `http://127.0.0.1:8888/object-links`。审核页按“数据集＋名词＋候选集合”合并重复项，一次人工决定会更新该数据集内全部相同映射，并仅重写镜像文件。页面支持查看对应 episode 的代表画面与所有相机视频、勾选一个或多个候选、搜索整个物体库、直接输入 ID，以及标记非物体。`tasks.jsonl` 是数据集级文本，没有精确帧时间，页面会明确标记其视频为代表 episode。

下载一个 RoboCOIN 数据集的第一条 episode，用于小规模测试：

```bash
python download_head_videos.py --limit 10
```

当前数据目录：

```text
RoboCOIN_datasets/
└── Galbot_G1_load_floor_trash_3/
    └── videos/chunk-000/
        ├── observation.images.cam_left_head_rgb/episode_000000.mp4
        └── observation.images.cam_right_head_rgb/episode_000000.mp4
```

下载更多数据时去掉 `--limit 1`，但建议先完成单数据集端到端测试。

## SAM 3 / SAM 3.1 权重

默认路径：

```text
sam3_weights/sam3.pt
sam3_weights/sam3.1_multiplex.pt
```

权重已从 ModelScope 的 `facebook/sam3` 模型仓库下载：

```text
文件大小：3,450,062,241 bytes
SHA-256：9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

已在本机 RTX 4070 Laptop 8 GB 上成功构建 `Sam3VideoPredictorMultiGPU`；只加载模型、尚未建立视频会话时，PyTorch 显存分配约 3.49 GiB。

## Stage 1：发现、审核、细化、跟踪

Stage 1 默认使用源视频第 0 帧发现物体。若单帧不能覆盖全部物体，可在人工审核页点击“追加关键帧”，拖动时间轴预览并确认；系统会在所选源帧重新运行 SAM3，同时保留此前关键帧及其物体。单帧自动发现使用 `object`，并补充数据集名称、`meta/tasks.jsonl` 和当前 episode 对应 `meta/episodes.jsonl` 的任务名词；不读取场景描述和 subtask。图像由 SAM3 编码一次，每个提示按照官方接口分别检测，最后合并并去除跨提示词重复掩码。Stage 1 强制拆成独立阶段，默认跟踪抽帧率为 1 FPS：

发现缓存即使包含人工审核 `revision`，也必须同时匹配当前缓存版本、提示词列表和提示策略；任一项过期都会重新运行 SAM3 并替换当前候选。旧目录仍会先写入 `.history/` 快照，但人工审核状态不会阻止算法升级后的全量重算。

首帧和人工确认的追加关键帧会由 RealESRGAN_x2plus 按比例增强到最长边 `2048` 像素，作为 SAM3 发现、机械臂检测、人工审核与框/点细化的统一图像。拖拽关键帧时的快速预览仍直接解码原视频，只在点击“确认并追加发现”后生成2K缓存。

SAM 3.1 跟踪仍使用原始抽帧；注入跟踪器前，会把2K审核掩码按最近邻缩放回原帧尺寸。选出最佳跟踪帧后，再将该帧超分到2K，并同步放大最佳掩码后导出 `best_quality.jpg`、上下文图和掩码。VLM 属性标注与 CLIP 去重因此会直接使用2K代表图。原图已达2K时直接复用。模型使用 CUDA FP16 和自适应分块，首次调用后常驻进程。可用 `ROBOCOIN_SUPER_RESOLUTION=0` 临时关闭，或用 `REALESRGAN_TILE=256` 手工指定分块边长。

```text
SAM3 使用 object + 数据名称中的名词发现，再检测并剔除机械臂/夹爪/机器人手（停止）
  → 人工新增/合并/删除
  → SAM3 按人工框细化
  → 再次人工审核
  → SAM 3.1 multiplex 使用当前保存的掩码以 1 FPS 跟踪
```

第一步只做首帧物体发现，完成后不会启动任何视频跟踪：

```bash
python stage1_track_select.py --stage discover --limit 1
```

自动发现首先使用通用 `object`，再加入明确的任务名词。例如 `move_test_tube` 可产生独立提示 `test tube`；文本中的单复数与重复名词会归一化，复合名词存在时不再重复发送其裸词头。语义名词进入 SAM3 之前会先由 VLM（`noun_review.py`，默认 GLM API 后端）审核，剔除明确的动词、动作、抽象概念和机械臂部位等非物体名词；审核请求会同时携带数据集名称和当前任务/场景标注上下文，并保守保留桌子、柜子等可分割实体及有歧义但可能是物体的词。判定按“名词、上下文、规则版本、API 地址和模型”缓存在 `objects/prompt_noun_review_cache.json`，避免不同场景或同名模型的不同服务错误复用。审核失败开放回退（fail-open）：API 不可用或返回异常时保留原始名词列表，不阻塞发现流程；`ROBOCOIN_NOUN_REVIEW=0` 可整体停用。跨提示掩码 IoU 达到 0.55，或包含率达到 0.80 且面积接近时视为重复。首帧发现与人工“确认并追加发现”使用同一组提示词。

自动发现阶段不再额外运行 `robot arm`、`robot gripper`、`robot hand` 提示；机械臂、背景和误检候选统一由人工审核删除。人工框细化仍可使用框及正负点精确排除机械臂和邻近物体。

自动首帧发现阈值为 0.2。低于阈值的候选不会保留，接触原图上、下、左、右任一边界的掩码会被硬过滤；如果机械臂或边界过滤移除了全部候选，也不会再强制恢复最高置信度候选。系统不再根据物体处于画面上方或面积较小自动删除、隐藏远景掩码，这些候选直接交给人工审核。某张关键帧允许没有合格掩码，可继续追加关键帧或人工框选后重新细化。人工框细化不使用自动发现的 0.2 阈值，也不应用原图边界过滤，而是使用 SAM3 原生交互分割头处理人工框和正负点。

人工确认新关键帧后会追加候选掩码，并可通过审核页的关键帧下拉框逐帧审核。每个物体记录自己的源视频帧号；跟踪抽帧时会强制插入所有发现关键帧，并在对应帧向 SAM 3.1 注入各自的审核掩码。重复物体可在审核阶段删除，或由后续 CLIP 聚簇与人工去重处理。

审核页可删除当前关键帧及其全部掩码，但每条数据至少保留一张关键帧。删除前自动建立撤销快照，图像文件不会立即物理删除。操作完成后页面会切换到相邻关键帧，并实时刷新各项数量和待跟踪状态。

修改后的人工矩形框可以通过审核页面的“SAM3 细化人工框”按钮处理，也可使用命令：

```bash
python stage1_track_select.py --stage refine --limit 1
```

细化会保留原始 `prompt_box`，把人工框直接作为 SAM3 box prompt；人工添加的正点和负点也会直接进入原生交互分割头并改变掩码结果。正点表示目标内部，负点用于排除背景、邻近物体或机械臂。模型默认返回三个候选掩码，在排除空掩码后按 SAM3 预测的掩码质量选择最佳结果。人工选择优先于自动机械臂判断，因此细化阶段不再执行机械臂重叠过滤。没有点时也可以只使用人工框。处理前会自动生成可撤销备份；细化后必须回到页面再次检查。

从审核页面执行细化时，SAM3 会在第一次请求时按需加载并常驻服务进程，后续细化直接复用，不再为每次点击重新加载权重。启动 SAM 3.1 跟踪或 VLM 建库任务前，服务会主动卸载常驻 SAM3 并清理 CUDA 缓存，以适配 8 GB 显存。直接运行 `--stage refine` 命令仍采用独立进程，不使用网页服务的模型缓存。

针对 8 GB GPU 和高分辨率鱼眼帧，项目版 SAM3 处理器会把候选掩码分成每批 8 个上采样回原图，并立即转为 CPU 布尔掩码；不会改变保存的原图尺寸或 SAM3 的 `1008×1008` 模型输入，只降低大量候选同时回映原尺寸时的显存峰值。网页服务还会在加载 torch 前启用可扩展 CUDA 分配段以降低碎片化风险。

满意后在页面点击“跟踪当前数据”，或者运行：

```bash
python stage1_track_select.py --stage track --limit 1
```

要忽略单条错误并跟踪所有已有掩码的数据，可以运行：

```bash
python stage1_track_select.py \
  --stage track \
  --continue-on-error
```

只跟踪审核后发生过掩码修改的数据，可以运行：

```bash
python stage1_track_select.py \
  --stage track \
  --continue-on-error \
  --dirty-only
```

审核窗口中的“一键跟踪全部待更新数据”执行这个增量流程。任何新增、合并、删除或重新细化都会把数据标记为“等待重新跟踪”；成功跟踪后自动恢复为“跟踪已更新”。未修改的数据不会重复抽帧、加载进跟踪器或改写输出，失败的数据仍保持待更新状态。

跟踪启动时会记录当前审核 `revision`。如果跟踪运行期间又发生人工修改，完成后的结果仍会保留，但清单继续显示“等待重新跟踪”，不会把新修改误标为已同步。跟踪结果先写入临时目录，全部图片和 manifest 成功生成后才整体替换 `tracker_sam3/`；中途取消、显存不足或写盘失败不会留下新旧混合结果，上一版完整目录归档在同数据目录的 `_tracker_archive/`。

指定单个视频和模型路径的完整写法：

```bash
python stage1_track_select.py \
  --stage track \
  --video /path/to/episode_000000.mp4 \
  --sam31-checkpoint /path/to/sam3.1_multiplex.pt \
  --sample-fps 1
```

输出按跟踪器分开，互不覆盖：

```text
objects/tracks/<video-key>/
├── first_frame/               # 默认第 0 帧及人工追加的多张关键帧
├── frames/                    # 从发现帧开始按 sample-fps 采样
├── initial_sam3_sr_2k/        # 2K 关键帧、SAM 3 掩码及 manifest
└── tracker_sam3/              # SAM 3.1 multiplex 掩码跟踪结果
```

每个跟踪结果中的物体目录包含：

```text
object_0000/
├── best_quality.jpg            # 综合质量最高的黑底物体裁剪
├── best_quality.png            # 人工树使用的无损黑底预览
├── best_quality_mask.png
├── best_quality_context.jpg
└── track.json                  # 每帧评分和选择结果
```

`tracker_sam3/representative_scenes/` 按代表帧共享保存2K完整场景。`track.json`
同时记录裁图和掩码的精确坐标，人工树查看完整背景时直接按坐标叠加，
不再使用模板匹配猜测位置。旧跟踪结果会先把2K裁图恢复到原始帧比例后再定位，
因此无需重新跟踪。

综合质量分数包括清晰度、掩码面积、SAM 置信度、相邻帧面积稳定性和物体是否完整位于画面内。默认权重依次为 `0.45 / 0.20 / 0.15 / 0.10 / 0.10`，运行跟踪时可以调整：

```bash
python stage1_track_select.py --stage track --tracker sam2 \
  --quality-sharpness 0.45 \
  --quality-area 0.20 \
  --quality-confidence 0.15 \
  --quality-stability 0.10 \
  --quality-boundary 0.10
```

权重不要求加起来等于 1，程序会自动归一化；必须均为非负数且不能全部为 0。每个候选帧的原始指标、最终分数以及本次使用的权重都会保存在 `track.json` 中。

也可以在 `/review` 页面的“代表图质量权重”区域直接修改这五项后再启动 SAM 3.1 分批跟踪。页面设置保存在浏览器本地，刷新后仍会保留；“恢复默认”可以还原为 `0.45 / 0.20 / 0.15 / 0.10 / 0.10`。

代表图只使用规则评分，不再调用 VLM 比较候选图，也不再生成 `quality_candidates/` Top-K 图片。每个物体只导出最终的 `best_quality.jpg`、掩码和场景上下文；全部帧的轻量评分元数据仍保存在 `track.json`，便于追踪选帧依据。

最终选择前会执行可调硬过滤：默认要求掩码面积至少达到该物体最大面积的25%、相邻帧面积稳定性至少0.50、排除清晰度最低10%的帧，并过滤接触画面边界的帧。如果某个物体所有帧都被过滤（例如机械臂始终延伸到画面外），系统会自动回退到全部帧，保证不会丢失物体。以上阈值和“过滤贴边帧”开关都可以在 `/review` 页面修改。

当前跟踪默认按 1 FPS 抽帧，并强制保留所有人工发现关键帧。若本地仍保留之前生成的高频抽帧缓存，下次运行 `track` 时会按当前 1 FPS 设置重建，同时插入所需关键帧。

## Stage 2 限制：amodal mask 与完整 RGB

`stage2_inpaint.py` 当前不能作为正式流水线运行，原因有两个：

1. 它把裁剪图中全部黑色背景当作待补区域，而正确区域应为 `amodal_mask - visible_mask`；
2. 它引用的 `moebius-inpainting` 和 `MoebiusPipeline` 并不是官方 Moebius 项目的安装/API方式。

正确流水线应为：

```text
visible_mask
    ↓ amodal segmentation
amodal_mask
    ↓ completion_mask = amodal_mask - visible_mask
RGB inpainting（只修改 completion_mask）
```

在接入 amodal segmentation 模型之前，不要运行现有 Stage 2，以免生成整块背景而误认为完整物体。虚拟环境已经安装 `diffusers==0.38.0`，为后续按官方 Moebius 实现重构做准备，但 Moebius 权重尚未下载。

## Stage 3：Qwen3-VL 属性提取

默认使用本地的 `models/Qwen3-VL-2B-Instruct/`。普通属性读取 `best_quality.jpg`；Category 还会把局部掩码自动定位回原始帧，提供高亮场景上下文。运行：

```bash
python stage3_attribute.py --tracker sam3
```

输出为 `objects/new_library_work/attributes.jsonl`。脚本只读取 SAM3 跟踪 manifest 中登记的实例，不读取仓库自带的 `objects/crops` 或旧物体库。重复运行按图片和提示词版本指纹复用缓存；`--force` 可强制重新标注。

属性标注默认以两个物体为一批执行，可通过 `--batch-size` 调整。场景上下文只保留目标附近约两倍范围，但不会缩放物体图或设置额外的像素上限；较大的目标仍使用较大的原始 ROI。如果一批大图超出显存，会自动清理 CUDA 缓存并逐个重试。对于没有具体掩码名称的 `object`，VLM 先只返回普通英文物体名；具体掩码名称则直接复用。本地 WordNet 先查询完整词组，再查询中心名词，只保留 physical entity 名词义项。唯一义项直接采用；多个义项由 VLM 根据图像、名称和定义返回候选编号；没有义项时从 `entity.n.01` 开始，每轮只展示当前节点的直接实体子节点并按编号下降。颜色、尺寸、材质、形状、纹理五项属性另行返回受控节点 ID。属性缓存每批原子写入一次。

保留五个属性字段：`category`、`color`、`material`、`shape`、`texture`，五项都不允许自由文本。受控词库位于 `ontology/attribute_taxonomies.json`：

- `category` 只允许沿 `entity.n.01 > physical_entity.n.01` 的实体分支导航，抽象分支不会进入人工树。无名称匹配时逐层选择可以继续进入 `object`、`substance`、`causal_agent` 等当前节点的直接实体子类。机械臂在 Stage 1 由 SAM3 检测后直接从候选掩码中剔除。
- 如果联合标注生成了不存在的 WordNet synset ID，会根据 `object_name_hint` 本地召回合法候选路径；VLM 只能返回候选编号，程序再从本地 WordNet 恢复路径。第一次选择失败时会扩大候选集合重试。
- 只有没有候选或两次候选编号选择都失败时才安全回退到 `entity.n.01` 并保存 `wordnet_warning`；人工调整树会显示橙色“WordNet待人工确认”提示。升级后只重新处理旧的 WordNet 警告缓存，其他有效属性缓存继续复用。
- `color`、`material`、`shape`、`texture` 分别沿自己的受控树逐层选择。例如 `color → green → dark green`。证据不足时停在父节点或选择 `unknown`。
- 缓存记录 `category_synset`、`category_path`、`attribute_paths` 和各轮原始判断，便于追踪和审核分类依据。

## Stage 4：属性优先、CLIP 聚簇与人工调整树

```bash
python stage4_dedup.py
python dedup_tree.py --regenerate
```

`stage4_dedup.py` 负责生成 ViT-L/14 嵌入；`dedup_tree.py` 按以下顺序生成可编辑树：

去重不再对候选对调用 VLM。Qwen 只负责每个物体的一次属性标注；随后按类别一致、受控属性相似度和 CLIP 视觉相似度生成候选，再由 complete-link 形成可编辑簇并交给人工审核。已有人工合并和拒绝决定继续保留。这样避免物体数量增加后产生大量成对 Qwen 推理。

候选缓存和最终物体库都保存来源指纹，指纹覆盖实例 ID、代表图、属性和分类路径。图片、跟踪结果或人工属性变化后，旧候选不能通过 `--reuse-candidates` 继续套用，旧物体库也不会再作为最新结果展示；需要重新运行完整去重或“补全属性并生成树”。人工去重决定只有在物体库重建成功后才提交，失败会自动回滚决定文件。

1. 先按照受控 `category_path` 建立属性树节点；
2. 同一节点内先检查 `color`、`material`、`shape`、`texture` 的兼容度；
3. 只有属性兼容的物体才使用 CLIP 视觉阈值聚簇；
4. 使用 complete-link 约束，避免仅靠相似链条把差异较大的物体串成大簇。

树的默认阈值可以通过命令或 `/dedup` 页面调整：

```bash
python dedup_tree.py --regenerate \
  --clip-threshold 0.82 \
  --attribute-threshold 0.50
```

`/dedup` 默认使用适合横屏的三层聚焦布局：上层显示当前类别的直接父类，中层显示当前类别及其同父兄弟类别，下层显示当前类别的直接子类。切换类别后这三层会以新节点为中心重新排布，层级从上向下、同层横向展开。节点卡片和右侧统计中的物体数、簇数只计算直接挂在当前节点的内容，不累加子节点。点击“显示全树”可切换到完整树总览，点击“显示当前节点三层”可返回三层聚焦布局。页面支持把物体拖到同节点或其他节点的已有簇，也可以拖到节点的“新建簇”区域进行拆分。每张物体卡的“查看完整背景”会显示完整原始帧，并用亮区和红框标出当前物体；VLM 属性推理仍使用目标附近的局部场景图。垃圾桶采用软删除：物体从树和新物体库移除，但原始跟踪和属性记录不会删除；垃圾桶中的物体可以拖回任意簇恢复。人工布局保存在 `dedup_tree_layout.json`，重新自动生成前会归档旧布局。

树布局同时保存每个物体的图像指纹。重新跟踪并生成树时，指纹未变的物体继承旧树的人工节点、聚簇和垃圾桶状态；指纹变化或新增的物体才使用新的属性与 CLIP 结果自动布局。审核掩码的物体 ID 为稳定 ID，删除其他掩码不会导致未变物体被重新编号。

主要输出包括：

```text
objects/new_library_work/clip_embeddings.npz
objects/new_library_work/dedup_tree_layout.json
objects/new_library_work/dedup_tree_archive/
objects/new_library/index.json
objects/new_library/<category>_<index>/
objects/object_id_registry.json
```

每个库对象的 `metadata.json` 都会保留 `source_instances`：其中包括来源数据路径、跟踪器、原物体编号、质量分、原图路径和当时的五项属性。合并只改变库中的组织关系，不删除任何原始跟踪图像。去重页面还可以直接重新运行 VLM 标注并建库。

最终物体库直接使用可读且持久的对象 ID：先把首次建库时的类别转为小写下划线形式，再在每个类别内从 0 开始永久分配，例如 `cup_0`、`cup_1`、`cup_2` 和 `mixing_bowl_0`。ID 注册表保存在 `objects/object_id_registry.json`；重建时通过实例重叠继承原 ID，实例增减、类别修正或对象删除都不会触发重编号，已经用过的编号也不会复用。合并时由重叠最强的一组继承已有 ID；拆分时原 ID 只由重叠最强的子组继承，其余子组获得新 ID。该 ID 同时作为目录名、API 主键和页面名称。

## 人工审核首帧物体

启动服务：

```bash
python viewer.py
```

打开审核窗口：

```text
http://127.0.0.1:8888/review
```

根地址 `http://127.0.0.1:8888/` 也会直接跳转到审核窗口，不会加载仓库自带的旧样例物体库。

页面会读取所有 `objects/tracks/**/initial_sam3_sr_2k/manifest.json`，可以切换视频并执行：

- 点击彩色掩码或右侧物体列表进行多选；
- 使用“框选新增”给漏检物体添加矩形初始掩码；
- 将错误拆成多个掩码的同一物体合并为一个并集掩码；
- 删除误检物体；
- 连续撤销人工操作；
- 给人工框添加原生正点或负点，并使用 SAM3 细化；
- 以2K原始像素绘制审核画布，支持适应窗口、1:1 原始2K和鼠标滚轮缩放；
- 确认掩码并启动 SAM 3.1 分批跟踪。

每次操作都会立即写回 `initial_sam3_sr_2k/`，操作前的 manifest 和掩码自动备份在对应目录的 `.history/` 中。框选新增首先得到矩形掩码，再由“SAM3 细化人工框”使用原生框和正负点提示生成贴合物体的掩码。SAM 3.1 跟踪完成后，页面状态会变为“跟踪已更新”。数据下拉框中的有效掩码数量会在修改和模型任务完成后自动刷新。

## 浏览物体库

这里只浏览由当前跟踪结果构建的 `objects/new_library`，不会读取仓库自带的 `objects/library` 和 `objects/crops`。

启动服务：

```bash
python viewer.py
```

浏览器打开：

```text
http://127.0.0.1:8888/new-library
```

去重审核页面：

```text
http://127.0.0.1:8888/dedup
```

物体详情页可以修改 VLM 属性，并查看每个实例来自哪段数据、原物体编号、质量分和当时的图像。聚簇树页面可以调整属性与 CLIP 阈值、拖拽物体跨簇或跨属性节点、拆分新簇，并通过垃圾桶软删除。每次人工操作都会同步重建物体库；前一版物体库和重新生成前的树布局都会归档。

修改 `category` 时，受影响实例会按新类别路径重新放置；其他实例继续继承已有人工节点、聚簇和垃圾桶状态。若属性保存或后续建库失败，属性覆盖文件会恢复到操作前版本。

## 推荐开发顺序

1. 用 SAM3 发现首帧物体；必要时人工追加关键帧，再逐帧修正并细化新增框；
2. 使用当前保存的掩码让 SAM 3.1 multiplex 以 1 FPS 跟踪；
3. 按可调质量评分只保留 `best_quality`；
4. 接入amodal segmentation，得到完整轮廓；
5. 按 `amodal_mask - visible_mask` 重构RGB补全；
6. 用 Qwen 对质量最高图做属性标注并人工修改；
7. 用 CLIP 召回相似实例，在 `/dedup` 人工确认后构建最终库。

## 效果评估报告

`reports/` 用于保存算法改动前后的可复现 A/B 结果。本次 SAM3 超分评估可重新运行：

```bash
python reports/compare_super_resolution.py
```

脚本配对读取 `initial_sam3/` 与 `initial_sam3_sr_2k/`，统计检出覆盖率、掩码数量、置信度和新旧掩码匹配 IoU。全部明显变化图永久保存在 `comparisons/all/`，同时从新增检出、丢失检出、数量增减和轮廓突变中分层选择少量代表样本放入 `comparisons/highlights/`。图中显示单个掩码置信度、新旧平均置信度及差值，匹配掩码使用相同颜色。`human_review.csv` 用于人工填写 `better`、`worse` 或 `same`；没有真值掩码时，候选数量增加只能说明召回发生变化，不能单独证明准确率提高。

默认报告把旧 `review_hidden` 候选当作普通掩码全部计入，并排除超分前或超分后 `revision > 0` 的整条数据，排除清单保存在 `excluded_manual_revisions.csv`。只有补充诊断需要时才使用 `--include-manual-revisions`。

数据集名称语义提示的可控 A/B 可重新运行：

```bash
python reports/compare_semantic_prompts.py
```

该实验对每张图只做一次超分和 SAM3 图像编码，从同一次运行拆出“仅 `object`”基线与“`object` + 数据集名词”结果；排除 `revision > 0` 的人工调整数据，所有中间掩码只写入独立报告目录，不覆盖 `initial_sam3_sr_2k/`、跟踪或人工树。结果保存在 `reports/2026-08-31_semantic_prompts_ab_v3/`，包括逐数据指标、全部新增候选对比图和精选图。由于没有真值，新增候选只表示召回机会，仍需人工判断是否为正确物体。v3 修正了无 WordNet 环境下英文复数词尾被过度裁剪的问题。

## 已知限制

- SAM 3.1 multiplex 在 8 GB 显存上的实际容量与速度仍取决于分辨率、帧数和物体数，大数据建议先用 `--limit 1` 验证；
- 8GB显存不适合直接运行Qwen3-VL-8B；
- Stage 2目前不是正确的amodal RGB补全实现；
- CLIP 相似度不能可靠判断细粒度同款物体，所以只召回候选、不自动合并；
- VLM 的类别、颜色、材质、形状和纹理均为视觉估计，必须人工复核；
- 首帧审核窗口尚无自由画笔和掩码拆分。

## VLM 属性标注后端：本地 Qwen3-VL 或 OpenAI 兼容 API

`stage3_attribute.py` 的属性标注推理可切换后端，WordNet 消歧和受控属性树校验始终在本地完成，两种后端写出的 `attributes.jsonl` 结构与缓存逻辑完全一致：

属性缓存指纹包含后端、API 地址和模型名。切换到 GLM API、更换 API 地址或修改 `VLM_MODEL` 后，旧 VLM 属性缓存会自动失效并重新标注，不需要额外使用 `--force`。

- `--vlm-backend local`（默认）：加载本地 `models/Qwen3-VL-2B-Instruct`（需要 `transformers`、`qwen-vl-utils` 与本地权重）。
- `--vlm-backend api`：调用 OpenAI 兼容的 `/chat/completions` 视觉接口（例如智谱 `glm-5.3-flash`），依赖已内置的 `vlm_backend.py`，无需 `transformers`/本地 Qwen 权重。命令：

```bash
export VLM_API_BASE=https://open.bigmodel.cn/api/paas/v4  # 必填
export VLM_API_KEY=<key>                                   # 必填
export VLM_MODEL=glm-5.3-flash                             # 可选，默认 glm-5.3-flash（注意连字符）
export ROBOCOIN_VLM_BACKEND=api            # 让 run_vlm_library_pipeline.py / viewer 任务默认走 API
.venv/bin/python stage3_attribute.py --vlm-backend api
```

可选环境变量：`VLM_API_TIMEOUT`（秒，默认 60）、`VLM_API_MAX_RETRIES`（默认 4，429/5xx/网络错误重试）、`VLM_API_CONCURRENCY`（批量并发请求数，默认 4）、`VLM_API_IMAGE_MAX_EDGE`（发送图片最长边像素，默认 1536）、`VLM_API_IMAGE_QUALITY`（JPEG 质量，默认 85）、`VLM_API_TEMPERATURE`（默认 0.0）、`VLM_API_REASONING_EFFORT`（`low`/`high`/`max`，默认 `low`，置空则不发送）、`VLM_API_THINKING`（如 `enabled`；`glm-5.3-flash` 只接受 `enabled`，置空则不发送）、`VLM_API_TOKEN_SCALE`（把调用方传的 token 预算乘上该系数再作为 `max_tokens`，默认 3，给强制深度思考模型预留推理 token）。发送前图片会按最长边等比缩小并转 JPEG，以控制带宽与费用；本地后端保持 2K 原图不压缩。

## 审核页中文名词的预翻译

`/review` 里的中文名词默认查 `noun_translations.py` 内置词表；为覆盖新数据集里出现的未知名词，可用轻量文本模型批量预翻译生成 `noun_translations_cache.json`（运行时不再调用 API，离线包同样可用）：

```bash
export VLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
export VLM_API_KEY=<key>
# 默认 glm-4.7-flashx（付费文本模型，实测约 1s/批，已自动关闭思考）；
# 可选 glm-4.6 / glm-4.5-airx / 免费 glm-4.7-flash：
#   --model glm-4.7-flashx
.venv/bin/python precompute_noun_translations.py
```

脚本默认用 4 个并发批次翻译（`--concurrency` 或环境变量 `VLM_TRANSLATE_CONCURRENCY` 可调），扫描所有 `objects/tracks/**/initial_sam3_sr_2k/manifest.json` 中出现的英文提示词，只翻译内置词表与已有缓存未覆盖的部分，结果按词合并写入缓存；词表（`NOUN_ZH`）仍优先于缓存。`export_offline_review.py` 导出离线包时会自动带上该缓存文件。
