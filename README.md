# STEAD: Structured Temporal Evidence-guided Anomaly Detection

STEAD is a research-oriented surveillance anomaly detection prototype. It turns
an event video into structured temporal evidence, asks a VLM/LLM reviewer for a
strict JSON review, and fuses rules plus model confidence into a graded alarm.

中文定位：结构化时序证据驱动的监控异常检测与分级报警框架。

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
  -> best.pt railway/track segmentation
  -> SAM2 segmentation + streaming-memory tracking when configured
  -> optical-flow temporal mask smoothing
  -> mask-IoU intrusion judgment
  -> sliding-window confirmation
  -> STEAD ROI/rule/VLM/alarm fusion
```

Branch `STEAD_SAM2` embeds the RailwayIntrusion_Tracking_SAM2 method inside
STEAD's detector/tracker path:

- detector adapter: YOLO11l person/cow/sheep detection
- track-region detector: `best.pt` railway/track segmentation
- tracker adapter: SAM2 video predictor with streaming memory
- temporal smoothing: optical-flow guided probability fusion
- rule source: mask-IoU plus object-overlap intrusion judgment and sliding-window confirmation

When SAMTracking produces a railway/track mask, STEP 04 uses mask-IoU
intrusion judgment as the rule source:

```text
object mask ∩ railway track mask
  -> max MaskIoU
  -> suspicious frame
  -> sliding-window confirmation
  -> sam_mask_iou_intrusion rule
```

If no segmentation/track mask is available, STEP 04 automatically falls back to
the polygon/line rules in the YAML config.

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
../GroundingDINO/fence_grounded_sam.py
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
python fence_grounded_sam.py --image weights/Fence.png --output outputs/fence_seg --text-prompt "entire continuous black metal chain link fence. full fence line. complete wire mesh barrier. long protective mesh fence. continuous guard net along the field boundary. fence panels and posts." --box-threshold 0.12 --text-threshold 0.12 --max-box-area-ratio 0.85
```

With ROI:

```bash
python fence_grounded_sam.py --image weights/Fence.png --output outputs/fence_seg --crop-roi 0,200,1920,1000 --text-prompt "entire continuous black metal chain link fence. full fence line. complete wire mesh barrier. long protective mesh fence. continuous guard net along the field boundary. fence panels and posts." --box-threshold 0.12 --text-threshold 0.12 --max-box-area-ratio 0.85
```

By default, `fence_grounded_sam.py` uses `--selection-mode best`, so
`mask_all.png`, `fence_grounded_sam_vis.jpg`, and YOLO-seg labels are generated
from the highest ranked full-fence candidate. Use `--selection-mode all` only
when you intentionally want to merge every kept GroundingDINO candidate.

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
  detection, SAM2 frame propagation, rail/track mask segmentation, optical-flow
  smoothing, mask-IoU judgment, and sliding-window confirmation. This is the
  closest mode to `RailwayIntrusion_Tracking_SAM2/outputs/test_annotated.mp4`.
- `--sam-temporal-mode fast`: SAM2 still advances frame by frame, but YOLO object
  detection and rail/track segmentation can run at `--sam-sample-every` and
  `--sam-track-mask-interval`, with cached detections/masks reused between
  intervals.

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
  --track sam_tracking \
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
  --track sam_tracking \
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
```

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
  python -m src.pipeline.analyze_event --input-type video  --video examples/test_h264.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/demo_image_sam_gpu --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --max-analysis-frames 0 --visualization-max-frames 0 --log-level DEBUG --sam-object-model weights/yolo11l.pt --sam-track-model weights/best.pt --sam-track-labels 1 --sam2-config sam2_hiera_l.yaml   --sam2-checkpoint weights/sam2_hiera_large.pt --track sam_tracking --sam-temporal-mode faithful --sam-imgsz 640 --sam-progress-interval 1 --sam-device cuda 

  python -m src.pipeline.analyze_event --input-type video --video examples/test_h264.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/sam_tracking_fast_check --vlm-provider mock --max-analysis-frames 60 --no-visualization --log-level INFO --sam-object-model weights/yolo11l.pt --sam-track-model weights/best.pt --sam-track-labels 1 --sam2-config sam2_hiera_l.yaml --sam2-checkpoint weights/sam2_hiera_large.pt --track sam_tracking --sam-temporal-mode fast --sam-sample-every 15 --sam-track-mask-interval 30 --sam-imgsz 640 --sam-progress-interval 1 --sam-device cpu

  python -m src.pipeline.analyze_event --input-type video --video examples/test1.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/stead_sam2_full --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --track sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/best.pt --sam-track-labels 1 --sam2-enabled true --sam2-config sam2_hiera_l.yaml --sam2-checkpoint weights/sam2_hiera_large.pt --sam2-scan-frames 30 --sam-temporal-mode faithful --sam-imgsz 640 --sam-device cuda --max-analysis-frames 300 --visualization-max-frames 300 



  # image
  python -m src.pipeline.analyze_event --input-type image --image examples/RailFence1.png --camera-id cam02 --rules configs/rules.image.yaml --output outputs/image_sam_tracking_rail --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --track sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/FenceRail.pt -sam-track-labels 1 --sam2-enabled true --sam2-config sam2_hiera_l.yaml --sam2-checkpoint weights/sam2_hiera_large.pt --sam-imgsz 640 --sam-device cuda

  python -m src.pipeline.analyze_event --input-type image --image examples/K88+300-1.png --camera-id cam02 --rules configs/rules.image.yaml --output outputs/K88+300-1 --vlm-provider qwen --vlm-timeout 60 --vlm-max-retries 2 --vlm-fallback-on-error true --track sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/FenceRail.pt --sam-track-labels 1  --sam2-enabled true --sam-imgsz 640 --sam-device cuda

  # Gemma

  ## image
  python -m src.pipeline.analyze_event --input-type image --image examples/RAIL_INTRUED.png --camera-id cam02 --rules configs/rules.image.yaml --output outputs/codeclean_test_img_1 --vlm_mode local  --vlm-local-endpoint http://localhost:8082/v1/chat/completions --vlm-local-model gemma-4-26B --vlm-timeout 600 --track sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/FenceRail.pt  --sam-track-labels 1 --sam-imgsz 640 --sam-device cuda
  ## video
  python -m src.pipeline.analyze_event --input-type video --video examples/test1.mp4 --camera-id cam02 --rules configs/rules.image.yaml --output outputs/codeclean_test --vlm-mode local --vlm-local-endpoint http://localhost:8082/v1/chat/completions --vlm-local-model gemma-4-26B --vlm-timeout 600 --vlm-max-retries 2 --vlm-fallback-on-error true --track sam_tracking --sam-object-model weights/yolo11l.pt --sam-track-model weights/FenceRail.pt --sam-track-labels 1 --sam2-enabled true --sam2-config sam2_hiera_l.yaml --sam2-checkpoint weights/sam2_hiera_large.pt --sam2-scan-frames 30 --sam-temporal-mode faithful --sam-imgsz 640 --sam-device cuda --max-analysis-frames 300 --visualization-max-frames 300
  
  python fence_grounded_sam.py --image weights/Fence2.png --output outputs/fence_seg_4 --box-threshold 0.12 --text-threshold 0.12 --max-box-area-ratio 0.85 --sam-candidate-count 12

  python fence_grounded_sam.py --image weights/Fence2.png --output outputs/fence_seg_continuous_fix --box-threshold 0.12 --text-threshold 0.12 --max-box-area-ratio 0.85 --sam-candidate-count 12 --mask-output-mode continuous-band --continuous-band-bins 64 --continuous-band-margin 8
  python fence_grounded_sam.py --image weights/Fence2.png --output outputs/fence_seg_linefit_1 --box-threshold 0.12 --text-threshold 0.12 --max-box-area-ratio 0.85 --sam-candidate-count 12 --mask-output-mode continuous-band --continuous-band-bins 64 --continuous-band-margin 4 --continuous-band-min-height-ratio 0.12 --continuous-band-max-height-ratio 0.32
  python fence_grounded_sam.py --image weights/Fence2.png --output outputs/fence_seg_linefit_2 --box-threshold 0.12 --text-threshold 0.12 --max-box-area-ratio 0.85 --sam-candidate-count 12 --mask-output-mode continuous-band --continuous-band-bins 64 --continuous-band-margin 2 --continuous-band-min-height-ratio 0.10 --continuous-band-max-height-ratio 0.26
```
