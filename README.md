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
  -> mock or Qwen/DashScope VLM review
  -> alarm_result.json
```

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

pip install -r requirements.txt
```

Python 3.10+ is recommended.

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
