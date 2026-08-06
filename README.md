# STEAD: Structured Temporal Evidence-guided Anomaly Detection

STEAD is a research-oriented surveillance anomaly detection prototype. It turns
an event video into structured temporal evidence, asks a VLM/LLM reviewer for a
strict JSON review, and fuses rules plus model confidence into a graded alarm.

Chinese summary: structured temporal evidence-guided surveillance anomaly
detection and graded alarm framework.
## Pipeline

```text
event video
  -> detector adapter
  -> tracker adapter
  -> ROI/rule engine
  -> keyframes + temporal windows
  -> event_evidence.json
  -> mock, local Gemma, or Qwen/DashScope VLM review
  -> alarm_result.json
```

Optional SAMTracking tracker mode:

```text
event video frames
  -> YOLO11l detection (person/cow/sheep)
  -> rule-region segmentation (YAML, sam_track, or guard_net)
  -> SAM2 segmentation + streaming-memory tracking when configured
  -> optical-flow temporal mask smoothing
  -> mask-IoU intrusion judgment
  -> sliding-window confirmation
  -> STEAD ROI/rule/VLM/alarm fusion
```

Branch `STEAD_SAM2` embeds the RailwayIntrusion_Tracking_SAM2 method inside
STEAD's detector/tracker path:

- detector adapter: YOLO11l person/cow/sheep detection
- rule-region detector: YAML ROI, `best.pt`/`FenceRail.pt`, or GuardNet
- tracker adapter: SAM2 video predictor with streaming memory
- temporal smoothing: optical-flow guided probability fusion
- rule source: mask-IoU plus object-overlap intrusion judgment and sliding-window confirmation

When SAMTracking produces a rule-region mask, STEP 04 uses mask-IoU
intrusion judgment as the rule source:

```text
object mask intersect rule-region mask
  -> max MaskIoU
  -> suspicious frame
  -> sliding-window confirmation
  -> mask_iou_intrusion rule
```

If no segmentation/track mask is available, STEP 04 automatically falls back to
the polygon/line rules in the YAML config.

Rule-region source can now be selected explicitly:

- `--rule-region-source yaml`: use YAML polygon/line ROI rules only.
- `--rule-region-source sam_track`: use the existing `--sam-track-model`
  railway/track/fence-rail segmentation mask as the mask-IoU rule region.
- `--rule-region-source guard_net`: use GroundingDINO + SAM2 guard-net
  segmentation, then optional `continuous-band` endpoint fitting, as the
  mask-IoU rule region.
- `--rule-region-source auto`: resolve to guard-net only when guard-net options
  are explicitly supplied; otherwise use `sam_track` when `--sam-track-model`
  is supplied, then fall back to YAML.

The resolved source is written to `sam_tracking_result.json.metadata` as
`rule_region_source_requested`, `rule_region_source_resolved`, and
`rule_region_available`.

This branch is STEAD-only. Legacy RTSP/FTP recording code has been removed from
this branch so the repository is focused on the research prototype.

## Project Layout

```text
src/
  alarm/        # final alarm fusion
  api/          # STEAD FastAPI app
  config/       # settings and config schemas
  evidence/     # EventEvidence schemas, windows, keyframes
  ingestion/    # event/video ingestion adapter shells
  perception/   # detector, tracker, segmentation adapters
  pipeline/     # command-line event analyzer
  rules/        # ROI geometry and rule engine
  storage/      # artifact helpers
  vlm/          # review providers, prompt, JSON validator
tests/
configs/
docs/
```

## Install

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# CUDA 13.0
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Python 3.10 or 3.11 is recommended for the full GroundingDINO/SAM2 stack. Python
3.12 can work in the current STEAD environment, but GroundingDINO editable
builds are more fragile. Python 3.13 is not recommended.

### SAM2

Install SAM2 into the same virtual environment:

```bash
cd ..
git clone https://github.com/facebookresearch/segment-anything-2.git sam2-main
cd sam2-main
python -m pip install -e .
python -m pip install decord

cd ../multi_camera_event_system
```

If this repository already has `../sam2-main`, install that existing copy
instead:

```bash
cd ../sam2-main
python -m pip install -e .
python -m pip install decord
cd ../multi_camera_event_system
```

Verify SAM2:

```bash
python -c "from sam2.build_sam import build_sam2; from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM2 OK')"
```

`decord` is required by the SAM2 video predictor. Image-only SAMTracking uses
YOLO object detection plus the rail/track segmentation model and disables SAM2
streaming memory, so missing `decord` only affects video-mode SAM2 propagation.

### GroundingDINO

GroundingDINO is used by the optional guard-net/fence open-vocabulary
segmentation script:

```text
../GroundingDINO/fence_grounded_sam_1.py
```

Install GroundingDINO into the same `multi_camera_event_system` virtual
environment. Use `python -m pip`, not bare `pip`, so the active venv is used:

```bash
cd ../../GroundingDINO

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-fixed.txt

# Important: install torch first, then disable build isolation for GroundingDINO.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e . --no-build-isolation
```

For CPU-only installation, replace the PyTorch command with:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify GroundingDINO:

```bash
python -c "from groundingdino.util.inference import load_model, predict; print('GroundingDINO OK')"
```

Common install issue:

```text
No module named pip
No module named torch
Getting requirements to build editable did not run successfully
```

Fix it by repairing `pip`, installing `torch` first, and running:

```bash
python -m pip install -e . --no-build-isolation
```

Run the guard-net/fence segmentation demo:

```bash
cd ../../GroundingDINO
python fence_grounded_sam_1.py --image weights/Fence.png --output outputs/fence_seg --text-prompt "entire continuous black metal chain link fence" --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 0.6
```

With ROI:

```bash
python fence_grounded_sam_1.py --image weights/Fence.png --output outputs/fence_seg --crop-roi 0,200,1920,1000 --text-prompt "entire continuous black metal chain link fence" --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 0.6
```

By default, `fence_grounded_sam_1.py` uses `--selection-mode best`, so
`mask_all.png`, `fence_grounded_sam_vis.jpg`, and YOLO-seg labels are generated
from the highest ranked full-fence candidate. Use `--selection-mode all` only
when you intentionally want to merge every kept GroundingDINO candidate.

The current script saves the selected raw SAM2 mask. The STEAD CLI still accepts
older continuous-band options for command compatibility, but records
`mask_output_mode: sam` in GuardNet metadata because `fence_grounded_sam_1.py`
does not perform continuous-band post-processing.

Typical command:

```bash
python fence_grounded_sam_1.py --image weights/Fence2.png --output outputs/fence_seg --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 0.6 --selection-mode best --target-area-ratio 0.55
```

Main outputs:

```text
outputs/<run>/mask_0.png                  # final selected SAM2 mask
outputs/<run>/mask_all.png                # merged final masks
outputs/<run>/fence_grounded_sam_vis.jpg  # overlay visualization
outputs/<run>/detections.json             # candidates, scores, selected indices, paths
outputs/<run>/labels/<image_stem>.txt     # YOLO-seg label when enabled
```

## Mock Demo

The mock path does not require GPU, a model file, or an external API key.

```bash
python -m src.pipeline.analyze_event \
  --input-type video \
  --video examples/demo.mp4 \
  --camera-id cam01 \
  --rules configs/rules.example.yaml \
  --output outputs/demo \
  --vlm-provider mock \
  --mock-detections
```

SAMTracking tracker adapter demo:

```bash
python -m src.pipeline.analyze_event \
  --input-type video \
  --video examples/test_h264.mp4 \
  --camera-id cam02 \
  --rules configs/rules.example.yaml \
  --output outputs/sam_tracking_demo \
  --vlm-provider mock \
  --tracker sam_tracking \
  --sam-object-model weights/yolo11l.pt \
  --sam-track-model weights/best.pt \
  --sam-track-labels 1 \
  --sam2-config sam2_hiera_l.yaml \
  --sam2-checkpoint weights/sam2_hiera_large.pt \
  --sam-temporal-mode faithful \
  --sam-iou-threshold 0.10 \
  --sam-window-size 5 \
  --sam-confirm-count 3 \
  --max-analysis-frames 60 \
  --visualization-max-frames 60
```

`--tracker simple_iou` remains the default. `--tracker sam_tracking` writes an
extra artifact:

```text
outputs/sam_tracking_demo/sam_tracking_result.json
```

If YOLO11/SAM2/`best.pt` weights are missing, the adapter degrades cleanly and
records the reason in `sam_tracking_result.json.metadata` instead of stopping
the STEAD pipeline.

SAMTracking video timing modes:

- `--sam-temporal-mode faithful` or `reference`: every frame runs YOLO object
  detection, SAM2 frame propagation, rule-region segmentation, optical-flow
  smoothing, mask-IoU judgment, and sliding-window confirmation. This is the
  closest mode to `RailwayIntrusion_Tracking_SAM2/outputs/test_annotated.mp4`.
- `--sam-temporal-mode fast`: SAM2 still advances frame by frame, but YOLO object
  detection and rule-region segmentation can run at `--sam-sample-every` and
  `--sam-track-mask-interval`, with cached detections/masks reused between
  intervals.

Optical-flow smoothing uses OpenCV Farneback by default and runs efficiently on
CPU; it does not require GPU. A future RAFT-style smoother would require GPU,
but the current STEAD path does not use RAFT.

Image input:

```bash
python -m src.pipeline.analyze_event \
  --input-type image \
  --image examples/alarm.jpg \
  --camera-id cam01 \
  --rules configs/rules.example.yaml \
  --output outputs/image_demo \
  --vlm-provider mock
```

Image runs use an image-only visualization path and write:

```text
outputs/image_demo/visualizations/annotated_image.jpg
outputs/image_demo/visualizations/visualization_summary.json
```

The annotated image overlays the configured ROI/rule geometry, detection boxes,
track IDs, and triggered rule status on the input image. Video runs keep the
original video visualization outputs.

Generated artifacts:

```text
outputs/demo/event_evidence.json
outputs/demo/vlm_review.json
outputs/demo/alarm_result.json
outputs/demo/pipeline.log
outputs/demo/visualizations/pipeline_overview.jpg
outputs/demo/visualizations/annotated_pipeline.mp4
outputs/demo/visualizations/evidence_animation.mp4
outputs/demo/visualizations/visualization_summary.json
```

Example `alarm_result.json`:

```json
{
  "event_id": "demo_event",
  "final_level": "high",
  "final_score": 0.846,
  "is_alarm": true,
  "rule_score": 0.9,
  "vlm_score": 0.78,
  "action": "immediate response"
}
```

## Qwen/DashScope

Copy the environment template:

```bash
copy .env.example .env
```

Set `DASHSCOPE_API_KEY` in `.env`, then run:

```bash
python -m src.pipeline.analyze_event \
  --video examples/demo.mp4 \
  --camera-id cam01 \
  --rules configs/rules.example.yaml \
  --output outputs/demo_qwen \
  --vlm-provider qwen
```

If no key is configured, use `--vlm-provider mock`.

## Local Gemma VLM

The CLI now supports two VLM access modes:

- `--vlm-mode web`: use Qwen/DashScope through the existing web API path.
- `--vlm-mode local`: use a LAN/local Gemma VLM exposed as an OpenAI-compatible `/v1/chat/completions` endpoint.

`--vlm_mode` is also accepted for compatibility with underscore-style CLI usage.

`--vlm-provider` selects the concrete reviewer:

- `auto`: choose `qwen` for `web`, choose `gemma` for `local`.
- `qwen`: valid only with `--vlm-mode web`.
- `gemma` / `local_gemma`: valid only with `--vlm-mode local`.
- `mock`: offline mock review; it ignores web/local access.

Conflicting combinations such as `--vlm-provider qwen --vlm-mode local` now fail
fast instead of silently switching provider.

Local Gemma default settings:

```text
endpoint: http://localhost:8082/v1/chat/completions
model: gemma-4-26B
authorization: Bearer sk-no-key-required
```

Example:

```bash
python -m src.pipeline.analyze_event \
  --input-type image \
  --image examples/RailFence3.png \
  --camera-id cam02 \
  --rules configs/rules.image.yaml \
  --output outputs/gemma_local \
  --vlm-provider gemma \
  --vlm-mode local \
  --vlm-local-endpoint http://localhost:8082/v1/chat/completions \
  --vlm-local-model gemma-4-26B \
  --vlm-timeout 600 \
  --tracker sam_tracking \
  --sam-object-model weights/yolo11l.pt \
  --sam-track-model weights/FenceRail.pt \
  --sam-track-labels 1 \
  --sam-imgsz 640 \
  --sam-device cuda
```

When `--vlm-provider` is omitted, the CLI uses `auto`: `web` resolves to Qwen and
`local` resolves to Gemma. Explicit `--vlm-provider mock` still forces offline
mock review.

For video input, Local Gemma uses the same structured temporal prompt as Qwen:
tracks, windows, ROI/rule triggers, and `metadata.sam_tracking.summary` from the
SAM2 mask-IoU pipeline. This keeps the video decision path aligned with the Qwen
API mode; only the VLM endpoint changes. For image input, the local path keeps a
smaller evidence packet by default to avoid local context overflow. The default
`--vlm-local-max-images` is `1` because small local context windows can be filled
by image tokens; raise it only when your local server has enough context length.
It writes:

```text
outputs/<event>/gemma_request_summary.json
outputs/<event>/gemma_raw_response.json
outputs/<event>/gemma_error.json   # only when the local request fails
outputs/<event>/vlm_review.json
```

Video example using the same SAMTracking flow as the Qwen command:

```bash
python -m src.pipeline.analyze_event \
  --input-type video \
  --video examples/test1.mp4 \
  --camera-id cam02 \
  --rules configs/rules.image.yaml \
  --output outputs/stead_sam2_local \
  --vlm-provider gemma \
  --vlm-mode local \
  --vlm-local-endpoint http://localhost:8082/v1/chat/completions \
  --vlm-local-model gemma-4-26B \
  --vlm-timeout 600 \
  --vlm-max-retries 2 \
  --vlm-fallback-on-error true \
  --tracker sam_tracking \
  --sam-object-model weights/yolo11l.pt \
  --sam-track-model weights/best.pt \
  --sam-track-labels 1 \
  --sam2-enabled true \
  --sam2-config sam2_hiera_l.yaml \
  --sam2-checkpoint weights/sam2_hiera_large.pt \
  --sam2-scan-frames 30 \
  --sam-sample-every 15 \
  --sam-track-mask-interval 30 \
  --sam-imgsz 640 \
  --sam-device cuda \
  --max-analysis-frames 300 \
  --visualization-max-frames 300
```

## ROI Rules

Edit `configs/rules.example.yaml`.

Supported rule types:

- `intrusion`: object center enters a polygon ROI
- `line_crossing`: a track crosses a configured line
- `loitering`: a track stays in an ROI longer than a threshold
- `gathering`: object count in an ROI exceeds a threshold
- `proximity`: person and equipment/object tracks are too close

## Visualization

By default, the CLI saves visual outputs for the first three STEAD stages:

- detector adapter: detection boxes with label and confidence
- tracker adapter: track IDs and trajectories
- ROI/rule engine: ROI polygons, line rules, triggered rule list

Files are written under:

```text
outputs/<event>/visualizations/pipeline_overview.jpg
outputs/<event>/visualizations/annotated_pipeline.mp4
outputs/<event>/visualizations/evidence_animation.mp4
outputs/<event>/visualizations/visualization_summary.json
```

`annotated_pipeline.mp4` is generated when the input video can be opened.
`evidence_animation.mp4` is generated from structured evidence, so it is also
available for mock demos or missing-video smoke tests.

Disable this with `--no-visualization`.

For long videos, the CLI limits processing by default:

- `--max-analysis-frames 900`
- `--visualization-max-frames 300`

Set either value to `0` to process the full video.

## SAMTracking Adapter

The optional SAMTracking adapter lives in `src/perception/sam_tracking_adapter.py`.
It is designed to reference the `RailwayIntrusion_Tracking_SAM2` method while
keeping STEAD runnable on CPU-only or dependency-light demo machines.

The adapter output contains:

- `tracks`: STEAD-compatible tracked detections for evidence building
- `frames`: per-frame detections, object-mask count, railway-mask availability,
  max mask IoU, suspicious flag, sliding-window count, and alarm flag
- `track_mask_contours` / `object_mask_contours`: compact mask outlines used by
  `annotated_pipeline.mp4`
- `intrusion_events`: confirmed sliding-window intrusion intervals
- `metadata`: model paths, loaded/degraded status, SAM2 status, thresholds, and
  fallback diagnostics

With `--tracker sam_tracking`, `annotated_pipeline.mp4` overlays:

- railway/track segmentation mask
- object mask outlines and detection boxes
- `MaskIoU`, sliding-window count, and `NORMAL/SUSPICIOUS/ALARM` status
- red border when the mask-IoU intrusion rule confirms an alarm

Useful CLI switches:

```text
--tracker simple_iou|sam_tracking
--rule-region-source yaml|sam_track|guard_net|auto
--sam-object-model <path-to-yolo11l.pt>
--sam-track-model <path-to-best.pt>
--sam2-config <path-to-sam2-config.yaml>
--sam2-checkpoint <path-to-sam2-checkpoint.pt>
--sam2-enabled true
--sam2-scan-frames 30
--sam2-prompt-mode bounding_box
--sam-iou-threshold 0.10
--sam-object-overlap-threshold 0.15
--sam-window-size 5
--sam-confirm-count 3
--sam-use-optical-flow true
--guard-net-box-threshold 0.12
--guard-net-text-threshold 0.12
--guard-net-max-box-area-ratio 0.85
--guard-net-candidate-count 12
--guard-net-mask-output-mode sam|continuous-band
--guard-net-continuous-band-margin 4
--guard-net-continuous-band-endpoint-source largest-component|all-mask
--guard-net-save-selected-sam-mask true|false
```

CLI 参数说明：

| 参数 | 说明 |
| --- | --- |
| `--input-type video/image` | 输入类型。`video` 走视频检测流程，`image` 走单图或图片目录批量流程。 |
| `--video <path>` | 单个视频路径；当路径是目录时，会按视频批处理运行。 |
| `--video-dir <dir>` | 视频批处理目录。会扫描目录内视频文件，并给每个视频创建独立输出子目录。 |
| `--image <path>` | 单张图片路径；当路径是目录时，会按图片批处理运行。 |
| `--recursive` | 批处理目录时递归扫描子目录。适用于 `--video-dir`、目录型 `--video`、目录型 `--image`。 |
| `--batch-limit <n>` | 批处理最多处理多少个文件。`0` 表示不限制。 |
| `--camera-id <id>` | 摄像头 ID，写入 evidence、日志和结果 JSON。 |
| `--rules <yaml>` | YAML 规则文件路径。普通 ROI 规则从这里读取；GuardNet 模式下仍保留作为 fallback 和配置入口。 |
| `--output <dir>` | 输出目录。单视频/单图直接写入该目录；批处理会在该目录下按文件名创建子目录，并生成 `batch_summary.json`。 |
| `--event-id <id>` | 可选事件 ID；不传时自动生成。批处理会为每个文件自动生成独立 ID。 |
| `--mock-detections` | 使用内置假检测结果，适合快速测试 CLI、日志、VLM 和输出结构，不调用真实检测模型。 |
| `--no-visualization` | 跳过可视化输出，减少运行时间。 |
| `--max-analysis-frames <n>` | 视频最多分析多少帧。`0` 表示完整视频；设置为正数时，SAM2 也只初始化对应长度的临时短视频，避免长视频占用过多内存。 |
| `--visualization-max-frames <n>` | 可视化视频最多覆盖原视频时间轴上的前多少帧。`0` 表示完整视频；GuardNet fast 模式会在这个时间范围内按 `--sam-sample-every` 同步抽帧写出，输出时长不会超过已覆盖的原视频时长。 |
| `--log-level DEBUG/INFO/WARNING/ERROR` | 日志等级。调试模型加载、规则来源、GuardNet fallback 时建议用 `INFO` 或 `DEBUG`。 |

VLM 参数：

| 参数 | 说明 |
| --- | --- |
| `--vlm-provider auto/mock/qwen/gemma/local_gemma` | VLM 提供方。`mock` 不访问模型；`qwen` 走线上 API；`gemma/local_gemma` 走本地 Gemma 兼容接口。 |
| `--vlm-mode web/local` | VLM 模式。`web` 对应线上 Qwen；`local` 对应局域网/本地 Gemma。 |
| `--vlm-local-endpoint <url>` | 本地 Gemma 的 OpenAI-compatible chat completions endpoint。 |
| `--vlm-local-model <name>` | 本地 Gemma 模型名，例如 `gemma-4-26B`。 |
| `--vlm-local-max-images <n>` | 传给本地 VLM 的最多图片数量，用于控制上下文长度。 |
| `--vlm-timeout <sec>` | VLM 请求超时时间。 |
| `--vlm-max-retries <n>` | VLM 请求失败后的最大重试次数。 |
| `--vlm-fallback-on-error true/false` | VLM 失败时是否生成 fallback review，避免整个 pipeline 中断。 |

检测、跟踪和 SAMTracking 参数：

| 参数 | 说明 |
| --- | --- |
| `--tracker simple_iou/sam_tracking` | 跟踪器。`simple_iou` 是 bbox IoU 简单跟踪；`sam_tracking` 使用 YOLO + SAM/SAM2 mask 跟踪与规则区 mask IoU 判定。 |
| `--rule-region-source yaml/sam_track/guard_net/auto` | 规则区域来源。`yaml` 使用规则文件 ROI；`sam_track` 使用 `--sam-track-model` 分割出的轨道/围栏 mask；`guard_net` 使用 GroundingDINO + SAM2 生成护网规则区；`auto` 自动选择。 |
| `--sam-object-model <pt>` | 目标检测 YOLO 模型，通常用于检测人，例如 `weights/yolo11l.pt`。GuardNet 模式下 SAM/SAM2 仍主要处理人/目标。 |
| `--sam-track-model <pt>` | `sam_track` 规则区域模型，例如轨道/围栏分割模型。`guard_net` 模式不使用这个参数。 |
| `--sam-track-labels <ids>` | 从 `--sam-track-model` 中选择哪些类别合并为规则区域 mask，例如 `1`。多个类别用逗号写，如 `1,2`。 |
| `--sam2-enabled true/false` | 是否启用 SAM2。开启后可使用 SAM2 streaming memory 进行视频 mask 推进。 |
| `--sam2-config <yaml>` | SAM2 配置文件路径。建议使用真实 SAM2 repo 里的路径，例如 `../sam2-main/sam2/configs/sam2/sam2_hiera_l.yaml`。 |
| `--sam2-checkpoint <pt>` | SAM2 checkpoint 路径，例如 `../sam2-main/checkpoints/sam2_hiera_large.pt`。 |
| `--sam2-scan-frames <n>` | 初始化 SAM2 prompt 时扫描前多少帧收集目标框。 |
| `--sam2-prompt-mode bounding_box/center_point/centroid` | 给 SAM2 的 prompt 类型。`bounding_box` 通常最稳定。 |
| `--sam-temporal-mode fast/faithful/reference` | 视频时序模式。`faithful/reference` 每帧运行目标检测并逐帧做 mask 判定；`fast` 可以间隔运行目标检测，但 SAM2 仍逐帧推进。GuardNet 规则区通常由参考帧生成，不需要每帧重新检测。 |
| `--sam-sample-every <n>` | `fast` 模式下每隔多少帧运行一次目标检测，主要影响人/目标检测频率。GuardNet fast 模式的 `annotated_pipeline.mp4` 也会按这个间隔同步抽帧写出。 |
| `--sam-track-mask-interval <n>` | `fast` 模式下每隔多少帧更新一次 `sam_track` 规则区 mask。GuardNet 规则区通常来自参考帧。 |
| `--sam-imgsz <n>` | YOLO 推理图片尺寸，例如 `640`。更大可能更准但更慢。 |
| `--sam-device cuda/cpu` | 模型运行设备。SAM2/YOLO 建议用 `cuda`；Farneback 光流平滑本身在 CPU 上运行。 |
| `--sam-iou-threshold <float>` | 目标 mask 与规则区域 mask 的 IoU 阈值，超过后判为 suspicious。 |
| `--sam-object-overlap-threshold <float>` | 目标 mask 中有多少比例落入规则区域的阈值，超过后判为 suspicious。 |
| `--sam-window-size <n>` | 连续帧确认窗口大小。 |
| `--sam-confirm-count <n>` | 窗口内 suspicious 帧数量达到该值后触发 alarm。 |
| `--sam-use-optical-flow true/false` | 是否启用光流平滑。当前 Farneback 光流在 CPU 上运行，不需要 GPU。 |
| `--sam-progress-interval <n>` | 每处理多少帧打印一次 SAMTracking 进度日志。 |

GuardNet 规则区域参数：

| 参数 | 说明 |
| --- | --- |
| `--guard-net-text-prompt <text>` | GroundingDINO 识别护网/围网的文本提示词。默认偏向 `chain link fence / wire mesh fence / protective net`。 |
| `--guard-net-box-threshold <float>` | GroundingDINO bbox 置信度阈值。检测不到护网时可尝试降低到 `0.10`。 |
| `--guard-net-text-threshold <float>` | GroundingDINO 文本匹配阈值。检测不到护网时可尝试降低到 `0.10`。 |
| `--guard-net-max-box-area-ratio <float>` | 过滤过大检测框。`0.85` 表示候选框面积超过图片 85% 时过滤。 |
| `--guard-net-candidate-count <n>` | 送入 SAM2 的候选框数量上限。更多候选可能覆盖更完整，但更慢。 |
| `--guard-net-crop-roi x1,y1,x2,y2` | 可选护网 ROI。先裁剪指定区域再检测，适合画面里草地、树、人干扰明显的场景。 |
| `--guard-net-mask-output-mode sam/continuous-band` | GuardNet 输出模式。`sam` 使用 SAM2 原始 mask；`continuous-band` 用护网 mask 端点拟合上下边界，生成连续护网带。 |
| `--guard-net-continuous-band-margin <px>` | `continuous-band` 上下边界外扩像素，避免护网边缘被切掉。 |
| `--guard-net-continuous-band-endpoint-source largest-component/all-mask` | 连续带端点来源。`largest-component` 只用最大连通域；`all-mask` 用整体 mask，更适合横向连续护网。 |
| `--guard-net-save-selected-sam-mask true/false` | 是否保存最终被选中的 SAM mask，便于检查 GuardNet 规则区生成质量。 |

## Logs

Each run writes step-by-step logs to both the console and:

```text
outputs/<event>/pipeline.log
```

The log covers:

- STEP 01 start/config
- STEP 02 detector/tracker
- STEP 03 evidence build
- STEP 04 ROI/rule engine
- STEP 05 temporal windows
- STEP 06 visualization
- STEP 07 VLM review
- STEP 08 alarm fusion
- STEP 09 artifact saving
- STEP 10 complete

Use `--log-level DEBUG|INFO|WARNING|ERROR` to adjust verbosity.

## API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8010
```

Endpoints:

- `GET /health`
- `POST /analysis/event`
- `GET /events`
- `GET /events/{event_id}/evidence`
- `GET /events/{event_id}/vlm-review`
- `GET /events/{event_id}/alarm`

## Tests

```bash
python -m compileall -q src tests
pytest -q
```

## Documentation

- `docs/STEAD_DESIGN.md`
- `docs/API.md`
- `docs/EXPERIMENT_PLAN.md`

## Security

Do not commit `.env`, API keys, camera accounts, RTSP passwords, private videos,
database files, or generated outputs. If a secret was ever committed in another
branch or shared externally, rotate it before deployment.


# COMMAND

```bash
  # video
  python -m src.pipeline.analyze_event --input-type video  --video examples/test_h264.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/demo_image_sam_gpu --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --max-analysis-frames 0 --visualization-max-frames 0 --log-level DEBUG --sam-object-model weights/yolo11l.pt --sam-track-model weights/best.pt --sam-track-labels 1 --sam2-config sam2_hiera_l.yaml   --sam2-checkpoint weights/sam2_hiera_large.pt --tracker sam_tracking --sam-temporal-mode faithful --sam-imgsz 640 --sam-progress-interval 1 --sam-device cuda

  python -m src.pipeline.analyze_event --input-type video --video examples/test_h264.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/sam_tracking_fast_check --vlm-provider mock --max-analysis-frames 60 --no-visualization --log-level INFO --sam-object-model weights/yolo11l.pt --sam-track-model weights/best.pt --sam-track-labels 1 --sam2-config sam2_hiera_l.yaml --sam2-checkpoint weights/sam2_hiera_large.pt --tracker sam_tracking --sam-temporal-mode fast --sam-sample-every 15 --sam-track-mask-interval 30 --sam-imgsz 640 --sam-progress-interval 1 --sam-device cpu

  python -m src.pipeline.analyze_event --input-type video --video examples/test1.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/stead_sam2_full --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/best.pt --sam-track-labels 1 --sam2-enabled true --sam2-config sam2_hiera_l.yaml --sam2-checkpoint weights/sam2_hiera_large.pt --sam2-scan-frames 30 --sam-temporal-mode faithful --sam-imgsz 640 --sam-device cuda --max-analysis-frames 300 --visualization-max-frames 300



  # image
  python -m src.pipeline.analyze_event --input-type image --image examples/RailFence1.png --camera-id cam02 --rules configs/rules.image.yaml --output outputs/image_sam_tracking_rail --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/FenceRail.pt --sam-track-labels 1 --sam2-enabled true --sam2-config sam2_hiera_l.yaml --sam2-checkpoint weights/sam2_hiera_large.pt --sam-imgsz 640 --sam-device cuda

  python -m src.pipeline.analyze_event --input-type image --image examples/K88+300-1.png --camera-id cam02 --rules configs/rules.image.yaml --output outputs/K88+300-1 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/FenceRail.pt --sam-track-labels 1  --sam2-enabled true --sam-imgsz 640 --sam-device cuda

  # Gemma

  ## image
  python -m src.pipeline.analyze_event --input-type image --image examples/RAIL_INTRUED.png --camera-id cam02 --rules configs/rules.image.yaml --output outputs/codeclean_test_img_1 --vlm_mode local  --vlm-local-endpoint http://localhost:8082/v1/chat/completions --vlm-local-model gemma-4-26B --vlm-timeout 600 --tracker sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/FenceRail.pt  --sam-track-labels 1 --sam-imgsz 640 --sam-device cuda
  ## video
  python -m src.pipeline.analyze_event --input-type video --video examples/test1.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/codeclean_test --vlm-mode local --vlm-local-endpoint http://localhost:8082/v1/chat/completions --vlm-local-model gemma-4-26B --vlm-timeout 600 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/FenceRail.pt --sam-track-labels 1 --sam2-enabled true --sam2-config sam2_hiera_l.yaml --sam2-checkpoint weights/sam2_hiera_large.pt --sam2-scan-frames 30 --sam-temporal-mode faithful --sam-imgsz 640 --sam-device cuda --max-analysis-frames 300 --visualization-max-frames 300

  python -m src.pipeline.analyze_event --input-type image --image examples\FenceRail --recursive --batch-limit 0 --camera-id cam02 --rules configs\rules.image.yaml --output outputs\FenceRail123 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights\yolo11l.pt --sam2-enabled true --sam2-config ..\sam2-main\sam2\configs\sam2\sam2_hiera_l.yaml --sam2-checkpoint ..\sam2-main\checkpoints\sam2_hiera_large.pt --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 1 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true

  # STEAD guard-net batch video mode. Each video gets an output subdirectory, and outputs/stead_guard_net_batch/batch_summary.json summarizes the batch.
  python -m src.pipeline.analyze_event --input-type video --video examples/SchoolFenceVideo/example --recursive --batch-limit 0 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/SchoolFenceVideo --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights/yolo11l.pt --sam-temporal-mode faithful --sam2-enabled true --sam2-config ../sam2-main/sam2/configs/sam2/sam2_hiera_l.yaml --sam2-checkpoint ../sam2-main/checkpoints/sam2_hiera_large.pt --sam2-scan-frames 100 --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 0.85 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true --max-analysis-frames 0 --visualization-max-frames 0

  # STEAD guard-net rule-region source
  python -m src.pipeline.analyze_event --input-type video --video examples/test1.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/stead_guard_net_full --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights/yolo11l.pt --sam2-enabled true --sam2-config ../sam2-main/sam2/configs/sam2/sam2_hiera_l.yaml --sam2-checkpoint ../sam2-main/checkpoints/sam2_hiera_large.pt --sam2-scan-frames 30 --sam-temporal-mode faithful --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 0.85 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true --max-analysis-frames 300 --visualization-max-frames 300

  python -m src.pipeline.analyze_event --input-type video --video examples/test2.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/stead_guard_net_full_8 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights/yolo11l.pt --sam2-enabled true --sam2-config ../sam2-main/sam2/configs/sam2/sam2_hiera_l.yaml --sam2-checkpoint ../sam2-main/checkpoints/sam2_hiera_large.pt --sam2-scan-frames 30 --sam-temporal-mode faithful --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 0.85 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true --max-analysis-frames 100 --visualization-max-frames 100

  python -m src.pipeline.analyze_event --input-type video --video examples/SchoolFenceVideo/ --camera-id cam02 --rules configs/rules.image.yaml --output outputs/stead_guard_net_full_10 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights/yolo11l.pt --sam2-enabled true --sam2-config ../sam2-main/sam2/configs/sam2/sam2_hiera_l.yaml --sam2-checkpoint ../sam2-main/checkpoints/sam2_hiera_large.pt --sam2-scan-frames 30 --sam-temporal-mode fast --sam-sample-every 5 --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 0.85 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true --max-analysis-frames 100 --visualization-max-frames 100

  python -m src.pipeline.analyze_event --input-type image --image examples\SchoolFence\Fence2.png --camera-id cam02 --rules configs\rules.image.yaml --output outputs\fence_seg_guard_net2 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights\yolo11l.pt --sam2-enabled true --sam2-config ..\sam2-main\sam2\configs\sam2\sam2_hiera_l.yaml --sam2-checkpoint ..\sam2-main\checkpoints\sam2_hiera_large.pt --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 0.85 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true

  python -m src.pipeline.analyze_event --input-type image --image examples\SchoolFence --recursive --batch-limit 0 --camera-id cam02 --rules configs\rules.image.yaml --output outputs\SchoolFence --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights\yolo11l.pt --sam2-enabled true --sam2-config ..\sam2-main\sam2\configs\sam2\sam2_hiera_l.yaml --sam2-checkpoint ..\sam2-main\checkpoints\sam2_hiera_large.pt --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 1 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true

    python -m src.pipeline.analyze_event --input-type image --image examples\SchoolFence\Fence5.png --camera-id cam02 --rules configs\rules.image.yaml --output outputs\fence_seg_denoise2 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights\yolo11l.pt --sam2-enabled true --sam2-config ..\sam2-main\sam2\configs\sam2\sam2_hiera_l.yaml --sam2-checkpoint ..\sam2-main\checkpoints\sam2_hiera_large.pt --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 0.85 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true --guard-net-mask-denoise true 

  # STEAD guard-net batch video mode. Each video gets an output subdirectory, and outputs/stead_guard_net_batch/batch_summary.json summarizes the batch.
  python -m src.pipeline.analyze_event --input-type video --video examples/SchoolFenceVideo/example/Fence_NorthGym2.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/test1 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights/yolo11l.pt --sam-temporal-mode fast --sam-sample-every 15 --sam2-enabled true --sam2-config ../sam2-main/sam2/configs/sam2/sam2_hiera_l.yaml --sam2-checkpoint ../sam2-main/checkpoints/sam2_hiera_large.pt --sam2-scan-frames 30 --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 0.85 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true --max-analysis-frames 0 --visualization-max-frames 0

  python -m src.pipeline.analyze_event --input-type video --video examples/SchoolFenceVideo/example --recursive --batch-limit 0 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/test2 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --tracker sam_tracking --rule-region-source guard_net --sam-object-model weights/yolo11l.pt --sam-temporal-mode fast --sam-sample-every 15 --sam2-enabled true --sam2-config ../sam2-main/sam2/configs/sam2/sam2_hiera_l.yaml --sam2-checkpoint ../sam2-main/checkpoints/sam2_hiera_large.pt --sam2-scan-frames 100 --sam-imgsz 640 --sam-device cuda --guard-net-box-threshold 0.12 --guard-net-text-threshold 0.12 --guard-net-max-box-area-ratio 0.85 --guard-net-candidate-count 12 --guard-net-mask-output-mode continuous-band --guard-net-continuous-band-margin 4 --guard-net-continuous-band-endpoint-source all-mask --guard-net-save-selected-sam-mask true --max-analysis-frames 0 --visualization-max-frames 0

  # GroundingDINO + SAM2 fence segmentation
  python fence_grounded_sam_1.py --image weights/Fence2.png --output outputs/fence_seg_raw_sam --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 0.6 --selection-mode best --target-area-ratio 0.55

  python fence_grounded_sam_1.py --image weights/Fence2.png --output outputs/fence_seg_all --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 0.6 --selection-mode all --target-area-ratio 0.55

  python fence_grounded_sam_1.py --image weights/Fence2.png --output outputs/fence_seg_roi --crop-roi 0,200,1920,1000 --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 0.6 --selection-mode best

  python fence_grounded_sam_1.py --image weights/Fence.png --output outputs/fence_seg_loose_area --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 1 --selection-mode best --target-area-ratio 0.55

  python fence_grounded_sam_1.py --image weights/Fence.png --output outputs/fence_seg_best_sam --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 1 --selection-mode best --target-area-ratio 0.55

  python fence_grounded_sam_1.py --image weights/Fence.png --output outputs/fence_seg_target_large --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 1 --selection-mode best --target-area-ratio 0.70

  python fence_grounded_sam_1.py --image weights/Fence.png --output outputs/fence_seg_old_like --box-threshold 0.15 --text-threshold 0.15 --max-box-area-ratio 1 --selection-mode all --target-area-ratio 0.55
```
