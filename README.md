# Multi-Camera Event Recording System / 多摄像头事件检测与录像系统

A multi-camera event detection and recording system. It monitors FTP alarm image folders for each camera, creates event records when alarms are triggered, and can output event videos, YOLO annotated results, and Qwen-VL multimodal analysis results.

多摄像头事件检测与录像系统。系统监听每路摄像机的 FTP 报警图片目录，在触发报警后自动生成事件记录，并可根据配置输出事件视频、YOLO 标注结果以及 Qwen-VL 多模态分析结果。

## Features / 功能概览

- Multi-camera support: each camera has independent RTSP, FTP alarm folder, and recording parameters.  
  多摄像机接入：每路摄像机独立配置 RTSP、FTP 报警目录和录像参数。
- Two analysis modes:  
  两种分析模式：
  - `video`: continuously captures RTSP streams into circular TS caches, then composes an MP4 containing pre-alarm and post-alarm footage.  
    `video`：持续拉取 RTSP，保存环形 TS 缓存；报警后合成包含报警前后片段的 MP4。
  - `image`: skips RTSP recording and analyzes FTP alarm images only.  
    `image`：不启动 RTSP 录像，仅分析 FTP 上传的报警图片。
- Smart event merging: repeated alarms from the same camera can extend the current event window.  
  智能事件合并：同一摄像机在 `post_seconds` 窗口内再次报警时自动延长事件。
- YOLO person detection for images or composed videos.  
  YOLO 人员检测：对图片或合成视频执行人员检测。
- Qwen-VL analysis through DashScope multimodal models.  
  Qwen-VL 分析：通过 DashScope 多模态模型生成异常事件文字分析。
- SQLite event database for metadata and analysis results.  
  SQLite 事件库：记录事件时间、图片/视频路径、分析结果、状态等元数据。
- Optional REST API for health checks, camera status, and event queries.  
  可选 REST API：提供健康检查、摄像机状态、事件列表和事件详情接口。
- Windows-friendly scripts and configuration templates.  
  Windows 友好：提供 `.bat`、`.ps1` 启动脚本和 Windows 配置模板。

## Workflow / 工作流程

```text
Camera RTSP / 摄像机 RTSP
  └─ video mode / video 模式
      └─ FFmpeg 1-second TS cache / FFmpeg 1 秒 TS 缓存

FTP alarm image / FTP 报警图片
  └─ watchdog listener / watchdog 监听
      └─ alarm event / 报警事件
          └─ scheduler / 事件调度器
              ├─ video mode: compose MP4 / video 模式：合成 MP4
              ├─ image mode: analyze image / image 模式：分析图片
              └─ YOLO + Qwen-VL analysis / YOLO + Qwen-VL 分析
                  └─ SQLite database / SQLite 数据库
```

## Project Structure / 目录结构

```text
multi_camera_event_system/
├─ app/                    # Application modules / 主程序模块
│  ├─ main.py              # Entry point / 程序入口
│  ├─ config.py            # YAML config loader / YAML 配置加载
│  ├─ recorder.py          # RTSP/FFmpeg cache recorder / RTSP/FFmpeg 缓存录像
│  ├─ event_listener.py    # FTP image folder listener / FTP 图片目录监听
│  ├─ scheduler.py         # Event state machine / 事件状态机
│  ├─ event_finalizer.py   # Video composition and finalization / 视频合成与事件收尾
│  ├─ ftp_image_detector.py# YOLO + DashScope/Qwen-VL analysis / YOLO + Qwen-VL 分析
│  ├─ database.py          # SQLite database access / SQLite 数据库访问
│  └─ api.py               # FastAPI REST API / FastAPI 接口
├─ configs/
│  ├─ cameras.windows.yaml # Windows config template / Windows 配置模板
│  └─ cameras.yaml         # Runtime config / 实际运行配置
├─ scripts/
│  ├─ run.bat              # Windows CMD launcher / Windows CMD 启动脚本
│  ├─ run.ps1              # Windows PowerShell launcher / PowerShell 启动脚本
│  └─ run.sh               # Linux/macOS launcher / Linux/macOS 启动脚本
├─ data/
│  ├─ cache/               # TS circular cache / TS 环形缓存
│  ├─ video/               # Event videos and annotated outputs / 事件视频与标注结果
│  ├─ db/                  # SQLite database / SQLite 数据库
│  └─ logs/                # Logs / 日志
├─ requirements.txt
├─ yolov8n.pt              # YOLO weights / YOLO 权重
└─ README.md
```

## Requirements / 环境要求

- Python 3.11+
- FFmpeg 4.2+; required for `video` mode.  
  FFmpeg 4.2+；`video` 模式必需。
- Network access to camera RTSP streams and FTP alarm image folders.  
  可访问摄像机 RTSP 地址和 FTP 报警图片目录。
- DashScope API Key if Qwen-VL analysis is required.  
  如需 Qwen-VL 分析，需要 DashScope API Key。
- Recommended OS: Windows 10/11 or Ubuntu 20.04+.  
  推荐系统：Windows 10/11 或 Ubuntu 20.04+。

## Installation / 安装

### Windows

```powershell
cd multi_camera_event_system
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ffmpeg -version
```

You can also run the setup wizard:  
也可以运行初始化脚本：

```cmd
scripts\setup-windows.bat
```

### Linux / macOS

```bash
cd multi_camera_event_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ffmpeg -version
```

## Configuration / 配置

Copy the template:  
复制配置模板：

```powershell
copy configs\cameras.windows.yaml configs\cameras.yaml
```

Linux/macOS:

```bash
cp configs/cameras.windows.yaml configs/cameras.yaml
```

Edit `configs/cameras.yaml`:  
编辑 `configs/cameras.yaml`：

```yaml
system:
  analysis_mode: "video"          # video | image
  save_annotated_images: true      # save YOLO annotated images in image mode
  base_cache_dir: "./data/cache"
  base_video_dir: "./data/video"
  db_path: "./data/db/events.db"
  log_dir: "./data/logs"
  ffmpeg_path: "ffmpeg"
  log_level: "INFO"
  scheduler_interval_sec: 1.0
  file_cleanup_interval_sec: 30.0
  enable_api: false
  api_host: "0.0.0.0"
  api_port: 8000

cameras:
  cam_001:
    name: "Camera 001"
    ip: "192.168.1.100"
    rtsp_url: "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
    ftp_image_dir: "D:/FTP/cam_001"
    event_type: "line_crossing"
    enabled: true
    pre_seconds: 5
    post_seconds: 10
    cache_seconds: 120
    cooldown_seconds: 3
    max_extend_seconds: 60
    output_dir: "./data/video/cam_001"
```

### Key Options / 常用配置说明

| Field / 字段 | Description / 说明 |
| --- | --- |
| `analysis_mode` | `video` caches RTSP and creates MP4; `image` analyzes FTP images only. / `video` 会缓存 RTSP 并生成 MP4；`image` 只分析 FTP 图片。 |
| `ftp_image_dir` | Local folder where the camera uploads alarm images. / 摄像机报警图片上传到本机的目录。 |
| `pre_seconds` | Seconds before the alarm to include in event video; only for `video` mode. / 事件视频包含报警前多少秒，仅 `video` 模式有效。 |
| `post_seconds` | Seconds to keep recording/waiting after the alarm. / 报警后继续等待或录像多少秒。 |
| `cache_seconds` | Circular cache duration; should be greater than `pre_seconds + post_seconds`. / 环形缓存时长，建议大于 `pre_seconds + post_seconds`。 |
| `cooldown_seconds` | Debounce interval for repeated alarms from the same camera. / 同一摄像机报警去抖间隔。 |
| `max_extend_seconds` | Maximum duration that one event can be extended. / 单个事件允许被连续报警延长的最大时长。 |

For Windows YAML paths, use `/` or double backslashes `\\`:  
Windows YAML 路径建议使用 `/` 或双反斜杠 `\\`：

```yaml
ftp_image_dir: "D:/FTP/cam_001"      # recommended / 推荐
ftp_image_dir: "D:\\FTP\\cam_001"    # also valid / 也可以
```

## DashScope / Qwen-VL

Create `.env` in the project root if multimodal analysis is required:  
如需多模态分析，在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

If the API key is missing, YOLO can still run, but Qwen-VL analysis will return an error or partial-success status.  
未配置 API Key 时，YOLO 仍可执行，但 Qwen-VL 分析会返回错误或部分成功状态。

## Running / 运行

### PowerShell

```powershell
.\scripts\run.ps1 -ConfigFile "configs\cameras.yaml" -LogLevel "INFO"
.\scripts\run.ps1 -ConfigFile "configs\cameras.yaml" -LogLevel "INFO" -AnalysisMode image
```

### CMD

```cmd
scripts\run.bat configs\cameras.yaml INFO video
scripts\run.bat configs\cameras.yaml INFO image
```

### Python

```bash
python -m app.main -c configs/cameras.yaml -l INFO --analysis-mode video
python -m app.main -c configs/cameras.yaml -l INFO --analysis-mode image
```

Disable REST API:  
不启用 REST API：

```bash
python -m app.main -c configs/cameras.yaml -l INFO --no-api
```

## Test an Alarm / 测试报警

Put an image into one camera's `ftp_image_dir` to trigger an event.  
向某一路摄像机的 `ftp_image_dir` 放入一张图片即可触发事件。

```powershell
python -c "from pathlib import Path; from PIL import Image; out=Path('D:/FTP/cam_001'); out.mkdir(parents=True, exist_ok=True); Image.new('RGB',(1280,720),'blue').save(out/'alarm_test.jpg')"
```

View logs:  
查看日志：

```powershell
Get-Content .\data\logs\event_system.log -Wait
```

## Outputs / 输出结果

- SQLite database: `data/db/events.db`  
  SQLite 数据库：`data/db/events.db`
- Log file: `data/logs/event_system.log`  
  日志文件：`data/logs/event_system.log`
- Event video in `video` mode:  
  `video` 模式事件视频：

```text
data/video/{camera_id}/{YYYY-MM-DD}/event_{camera_id}_{YYYYMMDD_HHMMSS}.mp4
```

- YOLO annotated video in `video` mode: usually `*_yolo.mp4` in the same folder.  
  `video` 模式 YOLO 标注视频：通常为同目录下的 `*_yolo.mp4`。
- YOLO annotated image in `image` mode: usually `*_yolo.jpg/png`.  
  `image` 模式 YOLO 标注图片：通常为 `*_yolo.jpg/png`。

## Query Database / 查询数据库

```bash
sqlite3 data/db/events.db "SELECT id,camera_id,media_type,status,event_time,video_path,image_path FROM events ORDER BY event_time DESC LIMIT 20;"
```

## REST API

When enabled, the default base URL is `http://localhost:8000`.  
启用 API 后默认地址为 `http://localhost:8000`。

| Endpoint / 接口 | Description / 说明 |
| --- | --- |
| `GET /health` | Health check / 健康检查 |
| `GET /status` | System status / 系统状态 |
| `GET /cameras` | Camera list / 摄像机列表 |
| `GET /cameras/{camera_id}` | Camera details / 单路摄像机详情 |
| `GET /events` | Event list; supports `camera_id` and `limit`. / 事件列表，支持 `camera_id`、`limit` 参数。 |
| `GET /events/{event_id}` | Event details / 事件详情 |
| `GET /events/active/all` | Active events / 当前活跃事件 |
| `GET /health/recorders` | Recorder health / RTSP 录像器健康状态 |

Examples / 示例：

```bash
curl http://localhost:8000/status
curl "http://localhost:8000/events?camera_id=cam_001&limit=20"
```

## Troubleshooting / 常见问题

### 1. `ffmpeg is not installed or not in PATH`

Install FFmpeg and add it to PATH, or specify the full path in config.  
安装 FFmpeg 并加入 PATH，或在配置中填写完整路径。

```yaml
ffmpeg_path: "C:/Program Files/ffmpeg/bin/ffmpeg.exe"
```

### 2. Alarm images are uploaded, but no events are created / 摄像机报警图片上传了，但系统没有事件

Check the following:  
检查以下项目：

- `ftp_image_dir` matches the real upload folder.  
  `ftp_image_dir` 是否和实际上传目录一致。
- Image extension is `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, or `.tiff`.  
  图片扩展名是否为 `.jpg/.jpeg/.png/.bmp/.gif/.tiff`。
- Camera is enabled with `enabled: true`.  
  当前摄像机是否 `enabled: true`。
- The event is not skipped by `cooldown_seconds`.  
  是否处于 `cooldown_seconds` 去抖时间内。
- Logs contain `Detected alarm event`.  
  日志中是否有 `Detected alarm event`。

### 3. `No ts files found` in video mode / video 模式提示 `No ts files found`

Check the following:  
检查以下项目：

- RTSP URL, username, and password are correct.  
  RTSP 地址、账号密码是否正确。
- FFmpeg is writing files into `data/cache/{camera_id}`.  
  FFmpeg 是否正在写入 `data/cache/{camera_id}`。
- `cache_seconds > pre_seconds + post_seconds`.  
  确认 `cache_seconds > pre_seconds + post_seconds`。
- System time is accurate.  
  系统时间是否准确。

### 4. Qwen-VL analysis fails / Qwen-VL 分析失败

Check `.env`:  
检查 `.env`：

```env
DASHSCOPE_API_KEY=your_key
```

Also make sure the machine can access DashScope upload and inference APIs.  
并确认机器可访问 DashScope 上传和推理接口。

### 5. YAML path errors on Windows / Windows YAML 路径报错

Do not use single backslashes:  
不要使用单反斜杠：

```yaml
ftp_image_dir: "D:\FTP\cam_001"   # wrong in YAML double quotes / 错误示例
```

Use one of the following:  
请使用以下写法：

```yaml
ftp_image_dir: "D:/FTP/cam_001"
# or / 或
ftp_image_dir: "D:\\FTP\\cam_001"
```

## Security Notes / 安全提示

- Do not commit real camera credentials, internal IP addresses, or DashScope API keys.  
  不要把真实摄像机账号、密码、内网地址、DashScope API Key 提交到 Git。
- Keep real configuration in local `configs/cameras.yaml` and `.env`, and exclude them with `.gitignore`.  
  建议将真实配置保存在本地 `configs/cameras.yaml` 和 `.env` 中，并通过 `.gitignore` 排除。
- The REST API has no built-in authentication. Use firewall, VPN, or reverse-proxy authentication before exposing it.  
  REST API 当前未做认证，如对外开放请使用防火墙、VPN 或反向代理鉴权。

## Version / 版本

- README updated: 2026-05-28  
  README 更新日期：2026-05-28
