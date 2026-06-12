# STEAD Design

STEAD means Structured Temporal Evidence-guided Anomaly Detection. The goal is
to turn a recorded surveillance event into reproducible evidence before asking a
large model for review.

## Architecture

Core research pipeline:

```text
event video/image
  -> detector adapter
  -> tracker adapter
  -> ROI rule engine
  -> keyframes + temporal windows
  -> event_evidence.json
  -> VLMReviewProvider
  -> alarm_engine.py
  -> alarm_result.json
```

This branch keeps only the STEAD research prototype. Live stream, FTP, or other
production ingestion systems should be connected through the adapter boundaries
under `src/ingestion/`.

## Schemas

`EventEvidence` contains event id, camera id, video path, time range, fps, ROI
rule triggers, object tracks, keyframes, windows, and metadata.

`VLMReview` is strict JSON: anomaly flag, event type, level suggestion,
confidence, evidence time, tracks, matched rules, reason, false-alarm flag, and
recommended action.

`AlarmResult` fuses rule score and VLM score into final level, score, reasons,
evidence references, uncertainty, and action.

## Rule Engine

`src/rules/rule_engine.py` supports intrusion, line crossing, loitering,
gathering, and proximity rules. Rules are configured by
`configs/rules.example.yaml`.

## Adapters

- YOLO: `src/perception/yolo_detector.py`
- Open vocabulary detector placeholder: `src/perception/open_vocab_detector.py`
- ByteTrack placeholder with Simple IOU fallback: `src/perception/bytetrack_adapter.py`
- SAM/SAM2 placeholder: `src/perception/sam_adapter.py`
- Qwen/DashScope provider: `src/vlm/qwen_provider.py`
- Offline mock provider: `src/vlm/mock_provider.py`

## Demo

```bash
python -m src.pipeline.analyze_event \
  --video examples/demo.mp4 \
  --camera-id cam01 \
  --rules configs/rules.example.yaml \
  --output outputs/demo \
  --vlm-provider mock \
  --mock-detections
```

Artifacts:

```text
outputs/demo/event_evidence.json
outputs/demo/vlm_review.json
outputs/demo/alarm_result.json
```

## Research Extension

For UCF-Crime, XD-Violence, ShanghaiTech, and UCSD Ped2, map each clip into
`EventEvidence`, evaluate frame-level/event-level labels with the same alarm
engine, and compare rule-only, VLM-only, caption+LLM, structured evidence+LLM,
and full STEAD variants.
