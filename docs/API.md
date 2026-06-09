# API

The legacy API remains in `app/api.py`.

The STEAD research API is in `src/api/main.py`.

Run:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8010
```

Endpoints:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/analysis/event` | Run STEAD pipeline |
| GET | `/events` | List output folders with alarm results |
| GET | `/events/{event_id}/evidence` | Return `event_evidence.json` |
| GET | `/events/{event_id}/vlm-review` | Return `vlm_review.json` |
| GET | `/events/{event_id}/alarm` | Return `alarm_result.json` |

Example body for `POST /analysis/event`:

```json
{
  "video_path": "examples/demo.mp4",
  "camera_id": "cam01",
  "rules_config": "configs/rules.example.yaml",
  "output": "outputs/api_event",
  "vlm_provider": "mock"
}
```

