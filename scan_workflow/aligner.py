"""
Step 1: Tray Aligner — ปรับแนวภาพด้วย Homography

Strategy: Option B+C (Overlay-guided + Reference Image)
    - ผู้ใช้เล็งกล้องตาม overlay ก่อนถ่าย (ลด deviation เบื้องต้น)
    - ใช้ SIFT feature matching กับ template reference image หา Homography matrix
    - Warp ภาพถ่ายให้อยู่บนมุมมองเดียวกับ reference image
    - slot bbox_norm ใน JSON จึงนำไปใช้ได้ตรงกับ aligned image ทันที

ความต้องการ:
    - template_ref_image: ภาพถาดที่ถ่ายตรง (เก็บใน admin_data/tray_refs/<tray_id>.jpg)
    - template JSON: มี bbox_norm และ image_size
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AlignResult:
    __slots__ = ("success", "aligned_image", "homography", "match_count", "message", "inlier_ratio")

    def __init__(
        self,
        success: bool,
        aligned_image: Optional[np.ndarray],
        homography: Optional[np.ndarray],
        match_count: int,
        inlier_ratio: float,
        message: str,
    ):
        self.success = success
        self.aligned_image = aligned_image
        self.homography = homography
        self.match_count = match_count
        self.inlier_ratio = inlier_ratio
        self.message = message

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "match_count": self.match_count,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "message": self.message,
            "has_homography": self.homography is not None,
        }


class TrayAligner:
    """
    ปรับแนวภาพถ่ายให้ตรงกับ template reference image

    กระบวนการ:
        1. ดึง keypoints จากทั้ง reference และภาพถ่าย (SIFT)
        2. Match features ด้วย FLANN + Lowe's ratio test
        3. คำนวณ Homography ด้วย RANSAC
        4. Warp ภาพถ่ายให้อยู่ในมุมมองเดียวกับ reference
    
    Fallback:
        ถ้า match ไม่พอ (<MIN_MATCHES) หรือ inlier ratio ต่ำ
        → ส่งคืน success=False พร้อม message ให้แจ้งผู้ใช้ถ่ายใหม่
    """

    MIN_MATCHES = 15          # จำนวน inlier ขั้นต่ำที่ยอมรับ
    MIN_INLIER_RATIO = 0.25   # สัดส่วน inlier/total matches ขั้นต่ำ

    def __init__(self, tray_refs_dir: str = "admin_data/tray_refs"):
        self.tray_refs_dir = Path(tray_refs_dir)

        # SIFT ปรับค่าสำหรับผิวโลหะ (เหมือนกับที่ scanner_module.py ใช้)
        self._sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=10)

        # FLANN matcher
        index_params = dict(algorithm=1, trees=5)  # algorithm=1 = FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Cache: เก็บ (ref_image, kp, des) ของแต่ละ tray_id
        self._ref_cache: dict[str, tuple[np.ndarray, list, np.ndarray]] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load_reference(self, tray_id: str) -> bool:
        """
        โหลด reference image สำหรับ tray_id เข้า cache
        
        ค้นหาไฟล์ใน: admin_data/tray_refs/<tray_id>.(jpg|jpeg|png)
        
        Returns:
            True ถ้าโหลดสำเร็จ
        """
        if tray_id in self._ref_cache:
            return True

        # ค้นหาไฟล์ reference
        ref_path = self._find_ref_image(tray_id)
        if ref_path is None:
            logger.warning(
                f"TrayAligner: ไม่พบ reference image สำหรับ '{tray_id}' "
                f"ใน {self.tray_refs_dir} — ไม่สามารถ align ได้"
            )
            return False

        img = cv2.imread(str(ref_path))
        if img is None:
            logger.error(f"TrayAligner: อ่านไฟล์ไม่ได้: {ref_path}")
            return False

        kp, des = self._extract_features(img)
        if des is None or len(kp) < self.MIN_MATCHES:
            logger.error(f"TrayAligner: reference image มี keypoint น้อยเกินไป: {len(kp)}")
            return False

        self._ref_cache[tray_id] = (img, kp, des)
        logger.info(f"TrayAligner: โหลด reference '{tray_id}' — {len(kp)} keypoints จาก {ref_path.name}")
        return True

    def set_reference_image(self, tray_id: str, image: np.ndarray) -> bool:
        """กำหนด reference image ใน memory โดยตรง"""
        if image is None or image.size == 0:
            return False
        kp, des = self._extract_features(image)
        self._ref_cache[tray_id] = (image.copy(), kp, des)
        logger.info(f"TrayAligner: set reference '{tray_id}' — {len(kp)} keypoints")
        return True

    def get_reference_image(self, tray_id: str) -> Optional[np.ndarray]:
        """ดึง reference image จาก cache (หรือโหลดถ้ายังไม่มี)"""
        if tray_id not in self._ref_cache:
            if not self.load_reference(tray_id):
                return None
        return self._ref_cache[tray_id][0]

    def align(self, query_image: np.ndarray, tray_id: str) -> AlignResult:
        """
        ปรับแนวภาพถ่ายให้ตรงกับ template reference

        Args:
            query_image : ภาพจากกล้องมือถือ (BGR)
            tray_id     : รหัสถาด เช่น "TRAY_YA_1_2"

        Returns:
            AlignResult — ถ้า success=True จะมี aligned_image และ homography
        """
        if not self.load_reference(tray_id):
            return AlignResult(
                success=False,
                aligned_image=None,
                homography=None,
                match_count=0,
                inlier_ratio=0.0,
                message=f"ไม่พบ reference image สำหรับถาด '{tray_id}' — กรุณาเพิ่ม reference image ในระบบ",
            )

        ref_img, ref_kp, ref_des = self._ref_cache[tray_id]
        h_ref, w_ref = ref_img.shape[:2]

        # ดึง features จากภาพถ่าย
        query_kp, query_des = self._extract_features(query_image)
        if query_des is None or len(query_kp) < self.MIN_MATCHES:
            return AlignResult(
                success=False,
                aligned_image=None,
                homography=None,
                match_count=len(query_kp),
                inlier_ratio=0.0,
                message="ภาพมี feature น้อยเกินไป กรุณาถ่ายให้เห็นถาดชัดและแสงสม่ำเสมอ",
            )

        # Match features
        good_matches, total_matches = self._match_features(ref_des, query_des)

        if len(good_matches) < self.MIN_MATCHES:
            return AlignResult(
                success=False,
                aligned_image=None,
                homography=None,
                match_count=len(good_matches),
                inlier_ratio=0.0,
                message=(
                    f"Match ได้ {len(good_matches)} จุด (ต้องการ ≥ {self.MIN_MATCHES}) "
                    f"— กรุณาเล็งให้เห็นถาดชัดกว่านี้"
                ),
            )

        # คำนวณ Homography
        H, mask, inlier_ratio = self._compute_homography(ref_kp, query_kp, good_matches)

        if H is None or inlier_ratio < self.MIN_INLIER_RATIO:
            return AlignResult(
                success=False,
                aligned_image=None,
                homography=None,
                match_count=len(good_matches),
                inlier_ratio=inlier_ratio,
                message=(
                    f"Homography ไม่เสถียร (inlier ratio {inlier_ratio:.0%}) "
                    f"— กรุณาถ่ายใหม่ให้ถาดเต็มเฟรม"
                ),
            )

        # Warp ภาพถ่ายให้ตรงกับขนาด reference
        aligned = cv2.warpPerspective(query_image, H, (w_ref, h_ref))

        inlier_count = int(mask.sum()) if mask is not None else len(good_matches)
        logger.info(
            f"TrayAligner: align สำเร็จ — {inlier_count} inliers, "
            f"ratio={inlier_ratio:.0%}, output={w_ref}×{h_ref}px"
        )

        return AlignResult(
            success=True,
            aligned_image=aligned,
            homography=H,
            match_count=inlier_count,
            inlier_ratio=inlier_ratio,
            message=f"ปรับแนวสำเร็จ ({inlier_count} จุดตรงกัน, {inlier_ratio:.0%} inliers)",
        )

    def get_reference_image(self, tray_id: str) -> Optional[np.ndarray]:
        """คืน reference image (BGR) สำหรับใช้สร้าง empty_ref crops"""
        if tray_id not in self._ref_cache:
            self.load_reference(tray_id)
        cached = self._ref_cache.get(tray_id)
        return cached[0] if cached else None

    def get_reference_size(self, tray_id: str) -> Optional[tuple[int, int]]:
        """คืน (width, height) ของ reference image"""
        img = self.get_reference_image(tray_id)
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _find_ref_image(self, tray_id: str) -> Optional[Path]:
        """ค้นหาไฟล์ reference image ของ tray_id"""
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            p = self.tray_refs_dir / f"{tray_id}{ext}"
            if p.exists():
                return p
        return None

    def _extract_features(self, image: np.ndarray) -> tuple[list, Optional[np.ndarray]]:
        """ดึง SIFT features จากภาพ (grayscale + CLAHE)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        # CLAHE ช่วยให้ feature ชัดขึ้นบนพื้นผิวโลหะสะท้อนแสง
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        kp, des = self._sift.detectAndCompute(gray, None)
        return kp, des

    def _match_features(
        self, des_ref: np.ndarray, des_query: np.ndarray, ratio: float = 0.75
    ) -> tuple[list, int]:
        """
        FLANN + Lowe's ratio test
        Returns: (good_matches, total_raw_matches)
        """
        try:
            matches = self._flann.knnMatch(des_ref, des_query, k=2)
        except cv2.error as e:
            logger.error(f"FLANN match error: {e}")
            return [], 0

        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)
        return good, len(matches)

    def _compute_homography(
        self, kp_ref: list, kp_query: list, good_matches: list
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        คำนวณ Homography matrix จาก matched keypoints ด้วย RANSAC
        Returns: (H, mask, inlier_ratio)
        """
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_query = np.float32([kp_query[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_query, pts_ref, cv2.RANSAC, ransacReprojThreshold=5.0)

        if H is None:
            return None, None, 0.0

        inlier_ratio = float(mask.sum()) / len(good_matches) if mask is not None else 0.0
        return H, mask, inlier_ratio
