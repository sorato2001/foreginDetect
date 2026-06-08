# Multi-Camera Railway Perimeter Event System / 多摄像头铁路周界事件系统

A Python demo for multi-camera railway perimeter intrusion recording and intelligent alarm review.

一个用于铁路周界入侵录像与智能报警复核的 Python Demo。

The system keeps the original capabilities:

系统保留原有能力：

- RTSP continuous capture with FFmpeg circular cache / 基于 FFmpeg 的 RTSP 循环缓存
- FTP alarm image trigger from Hikvision cameras / 海康摄像机 FTP 报警图片触发
- Pre-alarm and post-alarm event video generation / 报警前后事件视频生成
- SQLite event persistence / SQLite 事件入库
- FastAPI event query API / FastAPI 事件查询接口

It also adds a railway-security review pipeline:

同时新增铁路周界智能复核流程：

```text
RTSP cache / RTSP 缓存
  -> FTP alarm / FTP 报警
  -> YOLO detection / YOLO 目标初筛
  -> region rules / 区域规则
  -> Qwen-VL or mock VLM review / Qwen-VL 或 mock 多模态复核
  -> graded alarm explanation / 分级报警解释
  -> SQLite + FastAPI / SQLite 与 API 展示
```

## Features / 功能

- Multi-camera configuration / 多摄像机配置
- `video` mode: cache RTSP and compose alarm video / `video` 模式：缓存 RTSP 并合成报警视频
- `image` mode: analyze FTP alarm images only / `image` 模式：仅分析 FTP 报警图片
- YOLO detection for `person`, `vehicle`, and COCO animal classes / YOLO 检测人员、车辆和 COCO 动物类别
- Railway warning/danger zone rules / 铁路警戒区、危险区规则
- Fence-line crossing heuristic / 护网跨越启发式判断
- DashScope/Qwen-VL review with offline mock fallback / DashScope/Qwen-VL 复核，并支持离线 mock 降级
- Structured `analysis_result` saved to SQLite and returned by API / 结构化 `analysis_result` 入库并由 API 返回
- Demo scripts for image and event review / 图片与事件复核 Demo 脚本

## Project Structure / 目录结构

```text
multi_camera_event_system/
├─ app/
│  ├─ analysis/
│  │  ├─ yolo_detector.py      # YOLO wrapper / YOLO 封装
│  │  ├─ region_rules.py       # Railway region rules / 铁路区域规则
│  │  ├─ vlm_reviewer.py       # Qwen-VL/mock review / 多模态复核
│  │  ├─ frame_sampler.py      # Key-frame sampling / 关键帧抽取
│  │  └─ event_analyzer.py     # Unified analysis pipeline / 统一分析入口
│  ├─ main.py
│  ├─ scheduler.py
│  ├─ event_finalizer.py
│  ├─ event_listener.py
│  ├─ recorder.py
│  ├─ database.py
│  └─ api.py
├─ configs/
│  ├─ cameras.example.yaml
│  ├─ cameras.windows.yaml
│  └─ cameras.yaml
├─ scripts/
│  ├─ run.bat
│  ├─ run.ps1
│  ├─ run.sh
│  ├─ demo_analyze_image.py
│  └─ demo_analyze_event.py
├─ data/
├─ requirements.txt
├─ yolov8n.pt
└─ README.md
```

## Requirements / 环境要求

- Python 3.11+
- FFmpeg 4.2+ for `video` mode / `video` 模式需要 FFmpeg 4.2+
- Optional: DashScope API key for real Qwen-VL review / 可选：DashScope API Key 用于真实 Qwen-VL 复核
- Windows 10/11 or Linux / Windows 10/11 或 Linux

Install dependencies:

安装依赖：

```bash
cd multi_camera_event_system
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration / 配置

Copy the template:

复制模板：

```bash
cp configs/cameras.example.yaml configs/cameras.yaml
```

Windows CMD:

```cmd
copy configs\cameras.example.yaml configs\cameras.yaml
```

Basic camera settings:

基础摄像机配置：

```yaml
cameras:
  cam_001:
    name: "Railway Perimeter Camera 001"
    ip: "192.168.1.100"
    rtsp_url: "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
    ftp_image_dir: "./data/ftp/cam_001"
    event_type: "intrusion"
    enabled: true
    pre_seconds: 5
    post_seconds: 10
    cache_seconds: 120
    cooldown_seconds: 3
    max_extend_seconds: 60
    output_dir: "./data/video/cam_001"
```

### railway_security / 铁路周界智能复核配置

Add `railway_security` under each camera:

在每个摄像机下增加 `railway_security`：

```yaml
railway_security:
  enabled: true
  scene_type: "railway_perimeter"
  yolo:
    enabled: true
    model_path: "yolov8n.pt"
    conf_threshold: 0.35
    target_classes:
      - person
      - car
      - truck
      - bus
      - motorcycle
      - bicycle
      - dog
      - cat
      - horse
      - cow
      - sheep
  regions:
    image_size: [1920, 1080]
    fence_line:
      - [100, 620]
      - [1800, 610]
    warning_zone:
      - [80, 500]
      - [1850, 500]
      - [1900, 760]
      - [60, 760]
    danger_zone:
      - [150, 620]
      - [1800, 620]
      - [1860, 900]
      - [120, 900]
  risk_rules:
    min_persist_frames: 2
    low_risk: "目标在警戒区外经过"
    medium_risk: "目标靠近护网或进入警戒区"
    high_risk: "目标越过护网、进入危险区或向轨道方向移动"
  vlm:
    enabled: true
    provider: "dashscope"
    model: "qwen-vl-plus"
    timeout_sec: 20
    mock_when_no_key: true
```

`warning_zone` and `danger_zone` are polygons in image coordinates. `fence_line` is used to infer possible fence crossing. The system does not raise an alarm only because a person is detected; it combines target class, confidence, zone, and fence relationship.

`warning_zone` 和 `danger_zone` 是图像坐标多边形。`fence_line` 用于判断疑似跨越护网。系统不会因为“检测到人”就直接报警，而是结合目标类别、置信度、区域位置和护网关系进行判断。

## DashScope and Offline Mock / DashScope 与离线 Mock

For real Qwen-VL review, create `.env`:

如需真实 Qwen-VL 复核，在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

If no API key is configured, or DashScope/network is unavailable, the system automatically uses mock VLM review based on region rules.

如果未配置 API Key，或 DashScope/网络不可用，系统会自动使用基于区域规则的 mock 多模态复核，保证 Demo 可离线运行。

## Running / 运行

Windows PowerShell:

```powershell
.\scripts\run.ps1 -ConfigFile "configs\cameras.yaml" -LogLevel "INFO"
.\scripts\run.ps1 -ConfigFile "configs\cameras.yaml" -LogLevel "INFO" -AnalysisMode image
```

Windows CMD:

```cmd
scripts\run.bat configs\cameras.yaml INFO video
scripts\run.bat configs\cameras.yaml INFO image
```

Python:

```bash
python -m app.main -c configs/cameras.yaml -l INFO --analysis-mode video
python -m app.main -c configs/cameras.yaml -l INFO --analysis-mode image
```

## Demo Commands / Demo 命令

Analyze one alarm image:

分析单张报警图片：

```bash
python scripts/demo_analyze_image.py --image data/ftp/cam_001/alarm.jpg --camera-id cam_001 --config configs/cameras.yaml
```

Save visualization:

保存检测框、区域和风险等级可视化图片：

```bash
python scripts/demo_analyze_image.py --image data/ftp/cam_001/alarm.jpg --camera-id cam_001 --visualize data/demo_vis.jpg
```

Analyze a stored event:

分析数据库事件：

```bash
python scripts/demo_analyze_event.py --event-id 1 --config configs/cameras.yaml
```

Analyze a video file:

分析视频文件：

```bash
python scripts/demo_analyze_event.py --video data/video/cam_001/2026-05-28/event.mp4 --camera-id cam_001 --alarm-image data/ftp/cam_001/alarm.jpg
```

## REST API

Enable API in config:

在配置中启用 API：

```yaml
system:
  enable_api: true
  api_host: "0.0.0.0"
  api_port: 8000
```

Endpoints:

接口：

| Endpoint / 接口 | Description / 说明 |
| --- | --- |
| `GET /health` | Health check / 健康检查 |
| `GET /status` | System status / 系统状态 |
| `GET /cameras` | Camera list / 摄像机列表 |
| `GET /events` | Event list with alarm summary / 事件列表，包含报警摘要 |
| `GET /events/{id}` | Full event with `analysis_result` / 事件详情，包含完整复核结果 |
| `GET /events/summary/stats` | Risk-level statistics / 风险等级统计 |
| `GET /events/{id}/frames` | Key-frame paths / 关键帧路径 |
| `GET /health/recorders` | Recorder health / 录像器健康状态 |

Examples:

示例：

```bash
curl http://localhost:8000/events
curl http://localhost:8000/events/summary/stats
curl http://localhost:8000/events/1/frames
```

## analysis_result Example / analysis_result 示例

```json
{
  "camera_id": "cam_001",
  "event_time": "2026-05-28T10:00:00",
  "pipeline_version": "railway_security_v1",
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.82,
      "bbox": [520, 430, 610, 710],
      "center": [565, 570],
      "zone": "warning",
      "cross_fence": false
    }
  ],
  "rule_result": {
    "risk_level": "medium",
    "risk_score": 0.56,
    "reason": "人员进入警戒区"
  },
  "vlm_result": {
    "enabled": true,
    "is_valid_alarm": true,
    "risk_level": "medium",
    "target_type": "person",
    "behavior": "approaching",
    "reason": "人员靠近护网并进入警戒区域",
    "confidence": 0.72,
    "false_alarm_reason": ""
  },
  "final_result": {
    "is_alarm": true,
    "risk_level": "medium",
    "alarm_title": "铁路周界疑似入侵",
    "alarm_reason": "人员进入警戒区，建议现场核查",
    "recommended_action": "现场核查",
    "needs_review": false
  }
}
```

## Outputs / 输出

- Logs / 日志：`data/logs/event_system.log`
- Database / 数据库：`data/db/events.db`
- Event video / 事件视频：`data/video/{camera_id}/{YYYY-MM-DD}/event_*.mp4`
- Key frames / 关键帧：`*_frames/`
- Optional visualization / 可选可视化图：由 `--visualize` 指定

## Notes / 注意事项

- Do not commit real camera passwords or API keys. / 不要提交真实摄像机密码或 API Key。
- REST API has no built-in authentication. / REST API 当前未内置鉴权。
- VLM is only called on alarm images or sampled frames, never on the real-time stream. / 大模型只处理报警图片或抽帧，不处理实时视频流。
- If YOLO model or API key is missing, the system degrades gracefully. / 缺少 YOLO 模型或 API Key 时系统会降级运行。

## Version / 版本

- README updated: 2026-05-28
- Pipeline version: `railway_security_v1`
