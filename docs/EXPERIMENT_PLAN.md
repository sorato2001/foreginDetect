# Experiment Plan

## Datasets

- UCF-Crime
- XD-Violence
- ShanghaiTech
- UCSD Ped2
- Self-collected perimeter event clips

## Baselines

- Rule-only
- VLM-only
- Caption + LLM
- Structured Evidence + LLM
- Ours without trajectory
- Ours without rules
- Ours full

## Metrics

- frame-level AUC
- AP
- event-level F1
- false alarm rate
- miss rate
- average alarm delay
- p50/p95 latency
- VLM calls per event
- cost per event

## Ablations

- without trajectory
- without rules
- without VLM
- without keyframes
- different temporal window lengths
- different prompt templates
- different alarm thresholds

