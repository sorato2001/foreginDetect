"""Multimodal large-model reviewer for railway perimeter alarms.

DashScope/Qwen-VL is used when configured and available. The implementation
first tries DashScope's OpenAI-compatible HTTP API with a base64 image, so the
demo does not depend on the ``dashscope`` Python SDK. If no API key exists or
the real call fails, the reviewer returns a deterministic mock result based on
the region rules, keeping the demo runnable offline.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

VLM_JSON_SCHEMA_HINT = """
你是铁路周界安全报警复核助手。请根据图像、检测框信息和区域规则结果判断该报警是否有效。
场景：铁路周界/护网/轨道附近监控。
重点关注：
1. 是否有人、动物、车辆、施工机械、异物出现在画面中；
2. 目标是否靠近护网、进入警戒区、进入危险区或越过护网；
3. 是否存在攀爬、翻越、破坏护网、停留、向轨道方向移动等异常行为；
4. 是否可能是误报，例如光影、树枝、雨雪、画面噪声、远处正常经过人员；
5. 给出报警等级和简短原因。

必须严格返回 JSON：
{
  "is_valid_alarm": true/false,
  "risk_level": "normal|low|medium|high",
  "target_type": "person|animal|vehicle|object|unknown|none",
  "behavior": "passing|approaching|loitering|crossing|climbing|intruding|unknown",
  "reason": "不超过80字的中文解释",
  "confidence": 0.0-1.0,
  "false_alarm_reason": "如果不是有效报警，说明误报原因；否则为空字符串"
}
""".strip()


class VlmReviewer:
    """Review candidate alarm frames with DashScope/Qwen-VL or offline mock logic."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.provider = str(config.get("provider", "dashscope"))
        self.model = str(config.get("model", "qwen-vl-plus"))
        self.timeout_sec = float(config.get("timeout_sec", 20))
        self.mock_when_no_key = bool(config.get("mock_when_no_key", True))
        self.api_key_env = str(config.get("api_key_env", "DASHSCOPE_API_KEY"))

    def review(
        self,
        image_paths: list[str],
        detections: list[dict[str, Any]],
        rule_result: dict[str, Any],
        camera_id: str,
        event_time: str,
    ) -> dict[str, Any]:
        """Return VLM result in the unified schema."""
        if not self.enabled:
            return self._disabled_result(rule_result)

        api_key = os.getenv(self.api_key_env)
        if not api_key and self.mock_when_no_key:
            return self._mock_result(rule_result, detections, "未配置 DashScope API Key，使用离线 mock 复核。")
        if not api_key:
            return self._error_result("未配置 DashScope API Key", rule_result)

        if self.provider.lower() != "dashscope":
            return self._mock_result(rule_result, detections, f"暂不支持 provider={self.provider}，使用 mock 复核。")

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                self._review_with_dashscope,
                api_key,
                image_paths[:1],
                detections,
                rule_result,
                camera_id,
                event_time,
            )
            return future.result(timeout=self.timeout_sec)
        except FutureTimeoutError:
            logger.warning("VLM review timeout after %.1fs", self.timeout_sec)
            return self._mock_result(rule_result, detections, "VLM 复核超时，使用区域规则降级结果。")
        except Exception as exc:
            logger.error("VLM review failed: %s", exc, exc_info=True)
            return self._mock_result(rule_result, detections, f"VLM 复核失败，已降级：{exc}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _review_with_dashscope(
        self,
        api_key: str,
        image_paths: list[str],
        detections: list[dict[str, Any]],
        rule_result: dict[str, Any],
        camera_id: str,
        event_time: str,
    ) -> dict[str, Any]:
        """Call DashScope and parse strict JSON."""
        if not image_paths:
            return self._mock_result(rule_result, detections, "没有可供 VLM 复核的图片，使用规则结果。")

        prompt = self._build_prompt(detections, rule_result, camera_id, event_time)

        compatible_error = ""
        try:
            parsed = self._review_with_dashscope_compatible(api_key, image_paths[0], prompt)
            return self._normalize_result(parsed, rule_result)
        except Exception as compatible_exc:
            compatible_error = str(compatible_exc)
            logger.warning("DashScope compatible API failed, trying SDK/OSS path: %s", compatible_error)

        try:
            from ..ftp_image_detector import call_multimodal, upload_file_and_get_url

            temp_url = upload_file_and_get_url(api_key, self.model, image_paths[0])
            raw = call_multimodal(
                temp_url=temp_url,
                prompt=prompt,
                model=self.model,
                media_type="image",
                api_key=api_key,
            )
            if raw.get("status") != "success":
                sdk_msg = raw.get("msg") or raw.get("message") or "VLM 调用失败"
                return self._mock_result(
                    rule_result,
                    detections,
                    f"DashScope real call failed: compatible={compatible_error}; sdk={sdk_msg}",
                )
            parsed = self._parse_json(raw.get("answer", ""))
            parsed["enabled"] = True
            parsed["mode"] = "real"
            parsed["provider"] = self.provider
            parsed["model"] = self.model
            parsed["api"] = "dashscope_sdk_oss"
            return self._normalize_result(parsed, rule_result)
        except Exception as sdk_exc:
            return self._mock_result(
                rule_result,
                detections,
                f"DashScope real call failed: compatible={compatible_error}; sdk={sdk_exc}",
            )

    def _review_with_dashscope_compatible(
        self,
        api_key: str,
        image_path: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Call DashScope OpenAI-compatible multimodal API with a base64 image."""
        import base64
        import mimetypes

        import requests

        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as file_obj:
            image_b64 = base64.b64encode(file_obj.read()).decode("ascii")
        data_url = f"data:{mime};base64,{image_b64}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope compatible API failed: {response.status_code} {response.text[:500]}"
            )
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    text_parts.append(str(item))
            content = "\n".join(text_parts)
        parsed = self._parse_json(str(content))
        parsed["enabled"] = True
        parsed["mode"] = "real"
        parsed["provider"] = self.provider
        parsed["model"] = self.model
        parsed["api"] = "dashscope_compatible"
        return parsed

    def _build_prompt(
        self,
        detections: list[dict[str, Any]],
        rule_result: dict[str, Any],
        camera_id: str,
        event_time: str,
    ) -> str:
        context = {
            "camera_id": camera_id,
            "event_time": event_time,
            "detections": detections,
            "rule_result": rule_result,
        }
        return f"{VLM_JSON_SCHEMA_HINT}\n\n检测框信息和区域规则结果如下：\n{json.dumps(context, ensure_ascii=False)}"

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from model output, allowing accidental markdown fences."""
        if not text:
            raise ValueError("empty VLM response")
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            cleaned = match.group(0)
        return json.loads(cleaned)

    @staticmethod
    def _target_type_from_detections(detections: list[dict[str, Any]]) -> str:
        classes = {str(d.get("class_name", "")) for d in detections}
        if "person" in classes:
            return "person"
        if "animal" in classes or classes.intersection({"dog", "cat", "horse", "cow", "sheep"}):
            return "animal"
        if classes.intersection({"car", "truck", "bus", "motorcycle", "bicycle"}):
            return "vehicle"
        if classes:
            return "object"
        return "none"

    def _mock_result(
        self,
        rule_result: dict[str, Any],
        detections: list[dict[str, Any]],
        note: str,
    ) -> dict[str, Any]:
        risk = str(rule_result.get("risk_level", "normal"))
        target_type = self._target_type_from_detections(detections)
        is_valid = risk in {"medium", "high"}
        behavior = "intruding" if risk == "high" else "approaching" if risk in {"low", "medium"} else "passing"
        reason = str(rule_result.get("reason") or note)[:80]
        real_failed = "real call failed" in note.lower() or "\u590d\u6838\u5931\u8d25" in note or "timeout" in note.lower()
        return {
            "enabled": True,
            "mode": "mock",
            "mock": True,
            "real_call_failed": real_failed,
            "is_valid_alarm": is_valid,
            "risk_level": risk,
            "target_type": target_type,
            "behavior": behavior,
            "reason": reason,
            "confidence": 0.65 if is_valid else 0.5,
            "false_alarm_reason": "" if is_valid else "未发现进入警戒区或危险区的有效目标",
            "note": note,
        }

    @staticmethod
    def _disabled_result(rule_result: dict[str, Any]) -> dict[str, Any]:
        risk = str(rule_result.get("risk_level", "normal"))
        return {
            "enabled": False,
            "mode": "disabled",
            "is_valid_alarm": risk in {"medium", "high"},
            "risk_level": risk,
            "target_type": "unknown",
            "behavior": "unknown",
            "reason": "VLM 复核未启用，使用区域规则结果。",
            "confidence": float(rule_result.get("risk_score", 0.0)),
            "false_alarm_reason": "" if risk in {"medium", "high"} else "VLM 未启用且规则风险较低",
        }

    @staticmethod
    def _error_result(message: str, rule_result: dict[str, Any]) -> dict[str, Any]:
        risk = str(rule_result.get("risk_level", "normal"))
        return {
            "enabled": True,
            "mode": "error",
            "is_valid_alarm": risk in {"medium", "high"},
            "risk_level": risk,
            "target_type": "unknown",
            "behavior": "unknown",
            "reason": "VLM 复核不可用，使用区域规则结果。",
            "confidence": float(rule_result.get("risk_score", 0.0)),
            "false_alarm_reason": "" if risk in {"medium", "high"} else message,
            "error": message,
        }

    @staticmethod
    def _normalize_result(result: dict[str, Any], rule_result: dict[str, Any]) -> dict[str, Any]:
        allowed_risk = {"normal", "low", "medium", "high"}
        risk = str(result.get("risk_level") or rule_result.get("risk_level") or "normal")
        if risk not in allowed_risk:
            risk = str(rule_result.get("risk_level", "normal"))
        confidence = result.get("confidence", rule_result.get("risk_score", 0.0))
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except Exception:
            confidence = 0.0
        return {
            "enabled": bool(result.get("enabled", True)),
            "mode": str(result.get("mode", "real")),
            "is_valid_alarm": bool(result.get("is_valid_alarm", risk in {"medium", "high"})),
            "risk_level": risk,
            "target_type": str(result.get("target_type", "unknown")),
            "behavior": str(result.get("behavior", "unknown")),
            "reason": str(result.get("reason", "VLM 已完成复核"))[:80],
            "confidence": confidence,
            "false_alarm_reason": str(result.get("false_alarm_reason", "")),
            **({"provider": result["provider"]} if "provider" in result else {}),
            **({"model": result["model"]} if "model" in result else {}),
            **({"api": result["api"]} if "api" in result else {}),
        }
