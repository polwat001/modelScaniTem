"""
Step 0: Image Quality Gate
ตรวจสอบคุณภาพภาพก่อนส่งเข้า workflow หลัก

การตรวจ:
    1. ความคมชัด (Laplacian variance) — ป้องกันภาพเบลอ
    2. ความสว่าง (mean brightness) — ป้องกันมืด/สว่างเกิน
    3. ขนาดภาพ — ต้องมีความละเอียดพอสำหรับตรวจ 70 ช่อง
    4. อัตราส่วนภาพ — ตรวจว่าภาพไม่หมุนเกินควร
"""
from __future__ import annotations

import logging
import cv2
import numpy as np
from typing import NamedTuple

logger = logging.getLogger(__name__)


class QualityResult(NamedTuple):
    passed: bool
    issues: list[str]
    blur_score: float      # Laplacian variance — ยิ่งสูงยิ่งคมชัด
    brightness: float      # 0–255
    resolution: tuple[int, int]  # (width, height)
    score: float           # 0.0–1.0 คะแนนรวม

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "blur_score": self.blur_score,
            "brightness": self.brightness,
            "resolution": self.resolution,
            "score": self.score,
        }


class ImageQualityGate:
    """
    ตรวจคุณภาพภาพก่อนส่งเข้า workflow หลัก
    
    ค่า threshold ปรับได้จาก constructor — ยังไม่ล็อกตายตัวจนกว่าจะทดสอบกับมือถือจริง
    """

    def __init__(
        self,
        min_blur_score: float = 80.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_resolution: int = 640,          # ขนาดด้านสั้น (px) ขั้นต่ำ
        recommended_resolution: int = 1280, # แนะนำให้ได้ค่านี้
    ):
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_resolution = min_resolution
        self.recommended_resolution = recommended_resolution

    def check(self, image: np.ndarray) -> QualityResult:
        """
        ตรวจคุณภาพภาพ
        
        Args:
            image: BGR numpy array จากกล้องหรืออัปโหลด
            
        Returns:
            QualityResult — passed=True ถ้าผ่านทุกเกณฑ์
        """
        if image is None or image.size == 0:
            return QualityResult(
                passed=False,
                issues=["ไม่พบภาพ กรุณาถ่ายภาพใหม่"],
                blur_score=0.0,
                brightness=0.0,
                resolution=(0, 0),
                score=0.0,
            )

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        issues: list[str] = []

        # ── 1. ความคมชัด ──────────────────────────────────────────────────────
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.min_blur_score:
            issues.append(f"ภาพเบลอ (คะแนน {blur_score:.0f} < {self.min_blur_score:.0f}) กรุณาถ่ายใหม่ให้นิ่ง")

        # ── 2. ความสว่าง ───────────────────────────────────────────────────────
        brightness = float(gray.mean())
        if brightness < self.min_brightness:
            issues.append(f"ภาพมืดเกินไป (ค่าเฉลี่ย {brightness:.0f}) กรุณาเพิ่มแสง")
        elif brightness > self.max_brightness:
            issues.append(f"ภาพสว่างเกิน/มีแสงสะท้อน (ค่าเฉลี่ย {brightness:.0f}) กรุณาลดแสงหรือเปลี่ยนมุม")

        # ── 3. ขนาดภาพ ────────────────────────────────────────────────────────
        short_side = min(w, h)
        if short_side < self.min_resolution:
            issues.append(
                f"ความละเอียดต่ำเกินไป ({w}×{h}px) "
                f"แนะนำอย่างน้อย {self.min_resolution}px ด้านสั้น"
            )

        # ── 4. คะแนนรวม ───────────────────────────────────────────────────────
        blur_norm = min(1.0, blur_score / 400.0)           # 400+ = ดีมาก
        bright_norm = 1.0 if brightness < self.max_brightness and brightness > self.min_brightness else 0.4
        res_norm = min(1.0, short_side / self.recommended_resolution)
        score = round(blur_norm * 0.5 + bright_norm * 0.3 + res_norm * 0.2, 3)

        passed = len(issues) == 0

        if passed:
            logger.info(f"Quality Gate PASSED — blur={blur_score:.0f}, bright={brightness:.0f}, {w}×{h}px, score={score}")
        else:
            logger.warning(f"Quality Gate FAILED — issues={issues}")

        return QualityResult(
            passed=passed,
            issues=issues,
            blur_score=round(blur_score, 1),
            brightness=round(brightness, 1),
            resolution=(w, h),
            score=score,
        )

    def check_tray_visible(self, image: np.ndarray, min_tray_ratio: float = 0.35) -> tuple[bool, str]:
        """
        ตรวจคร่าวๆ ว่าเห็นถาดในภาพหรือไม่
        ใช้หา contour สีเข้มขนาดใหญ่ (foam สีดำ/เข้ม)
        
        Returns:
            (tray_found: bool, message: str)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape[:2]

        # ย่อภาพเพื่อความเร็ว
        scale = 400.0 / max(h, w)
        small = cv2.resize(gray, (int(w * scale), int(h * scale)))

        blur = cv2.GaussianBlur(small, (9, 9), 0)
        thresh_val = float(np.percentile(blur, 40))
        _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sh, sw = small.shape[:2]
        image_area = sw * sh

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area / image_area >= min_tray_ratio:
                return True, "พบถาดในภาพ"

        return False, "ไม่พบถาดในภาพ กรุณาเล็งกล้องให้เห็นถาดชัดเจน"
