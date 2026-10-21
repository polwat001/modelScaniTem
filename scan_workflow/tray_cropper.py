"""
โมดูล TrayBoxCropper — ตรวจจับขอบเขตถาดเครื่องมือ (Tray Boundary Detection)
ตรวจสอบระยะห่างการถ่ายภาพ (Distance Guard) และตัดขอบพร้อมปรับระนาบ (Auto-Crop & Perspective Rectification)
"""

from typing import NamedTuple, Optional, Tuple
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TrayCropResult(NamedTuple):
    """ผลลัพธ์จากการตรวจจับและตัดภาพถาด"""
    success: bool
    cropped_image: Optional[np.ndarray]
    corners: Optional[np.ndarray]          # 4 จุดมุม [TL, TR, BR, BL] ในพิกัดภาพเดิม
    tray_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) ในภาพเดิม
    area_ratio: float                      # สัดส่วนพื้นที่ถาดเทียบกับภาพทั้งหมด (0.0 - 1.0)
    is_too_far: bool                       # True ถ้าถ่ายไกลเกินไป (area_ratio < min_tray_ratio)
    message: str

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "has_crop": self.cropped_image is not None,
            "corners": self.corners.tolist() if self.corners is not None else None,
            "tray_bbox": list(self.tray_bbox) if self.tray_bbox is not None else None,
            "area_ratio": round(self.area_ratio, 3),
            "is_too_far": self.is_too_far,
            "message": self.message,
        }


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    จัดลำดับ 4 จุดมุมให้อยู่ในลำดับ:
    0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # ผลบวก x + y: ค่าน้อยสุด = Top-Left, ค่ามากสุด = Bottom-Right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # ผลต่าง y - x (หรือ x - y):
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # x มาก y น้อย -> Top-Right
    rect[3] = pts[np.argmax(diff)]  # x น้อย y มาก -> Bottom-Left
    
    return rect


class TrayBoxCropper:
    """
    ตรวจจับขอบถาดโฟมดำ (Dark Foam Tray)
    ตัดขอบโต๊ะ/พื้นหลังออก และทำ Perspective Transform ปรับแนวระนาบ
    """

    def __init__(
        self,
        min_tray_ratio: float = 0.35,      # สัดส่วนพื้นที่ถาดขั้นต่ำ (ถ้าต่ำกว่านี้ถือว่าถ่ายไกลเกินไป)
        optimal_tray_ratio: float = 0.50,  # สัดส่วนที่แนะนำ
        dark_thresh_percentile: float = 40.0, # สำหรับ threshold ค้นหาโฟมดำ
    ):
        self.min_tray_ratio = min_tray_ratio
        self.optimal_tray_ratio = optimal_tray_ratio
        self.dark_thresh_percentile = dark_thresh_percentile

    def detect_and_crop(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = (1448, 1086), # (width, height)
    ) -> TrayCropResult:
        """
        ตรวจจับขอบเขตถาด ตัดขอบพื้นหลัง และทำ Perspective Warp
        
        Args:
            image       : ภาพจากกล้องหรืออัปโหลด (BGR)
            target_size : ขนาดผลลัพธ์ (กว้าง, สูง) หรือ None เพื่อคงขนาดตาม bounding rect
            
        Returns:
            TrayCropResult
        """
        if image is None or image.size == 0:
            return TrayCropResult(
                success=False,
                cropped_image=None,
                corners=None,
                tray_bbox=None,
                area_ratio=0.0,
                is_too_far=True,
                message="ภาพไม่ถูกต้องหรือเป็นภาพว่าง",
            )

        h_orig, w_orig = image.shape[:2]
        total_img_area = float(h_orig * w_orig)

        # 1. ย่อภาพเพื่อหา Contour ได้รวดเร็วและเสถียร (max dim 600px)
        scale = 600.0 / max(h_orig, w_orig)
        small = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # 2. ค้นหาโฟมดำของถาดเครื่องมือ (Dark Foam Segmentation)
        # ถาดโฟม EVA ส่วนใหญ่มีค่าเฉลี่ย grayscale มืด (< 110)
        # ตรวจจับด้วย Absolute dark threshold ผสมกับ percentile เพื่อความเสถียร
        p_val = float(np.percentile(blur, self.dark_thresh_percentile))
        thresh_val = min(115.0, max(50.0, p_val))
        _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

        # ค้นหา Contour ที่ใหญ่ที่สุด
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        max_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0.10 * (sw * sh) and area > max_area:
                # ตรวจสอบว่าทรงใกล้เคียงสี่เหลี่ยม (Solidity > 0.65)
                bx_t, by_t, bw_t, bh_t = cv2.boundingRect(cnt)
                solidity = area / float(max(1, bw_t * bh_t))
                if solidity > 0.60:
                    max_area = area
                    best_cnt = cnt

        # 3. Fallback: ถ้าไม่เจอด้วย Dark Threshold ให้ลองใช้ Canny Edge Contours
        if best_cnt is None:
            edges = cv2.Canny(blur, 40, 120)
            edges_dil = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
            cnts2, _ = cv2.findContours(edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts2:
                area = cv2.contourArea(cnt)
                if area > 0.15 * (sw * sh) and area > max_area:
                    max_area = area
                    best_cnt = cnt

        # 4. หากยังไม่พบขอบถาดที่ชัดเจน (อาจเพราะผู้ใช้ถ่ายเต็มจอพอดีขอบภาพอยู่แล้ว)
        if best_cnt is None:
            # ใช้ภาพทั้งภาพ ถือว่าถาดเต็มจอ 100%
            out_img = image
            if target_size is not None:
                tw, th = target_size
                # สลับมิติถ้าภาพกับ target_size มีทิศทางตรงกันข้าม เพื่อรักษาสัดส่วน
                if (w_orig > h_orig and th > tw) or (h_orig > w_orig and tw > th):
                    tw, th = th, tw
                out_img = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            return TrayCropResult(
                success=True,
                cropped_image=out_img,
                corners=np.array([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]], dtype="float32"),
                tray_bbox=(0, 0, w_orig, h_orig),
                area_ratio=1.0,
                is_too_far=False,
                message="ไม่พบขอบนอกถาดแยกจากภาพ (ภาพอาจถูกถ่ายเต็มเฟรมอยู่แล้ว)",
            )

        # สเกล contour กลับเป็นพิกเซลขนาดภาพจริง
        cnt_orig = (best_cnt / scale).astype(np.int32)
        bx, by, bw, bh = cv2.boundingRect(cnt_orig)
        tray_area = float(cv2.contourArea(cnt_orig))
        area_ratio = tray_area / total_img_area

        # ตรวจสอบว่าถ่ายไกลเกินไปหรือไม่
        is_too_far = area_ratio < self.min_tray_ratio

        # 5. สกัดจุด 4 มุมของถาด (4-Corner Detection)
        peri = cv2.arcLength(cnt_orig, True)
        approx = cv2.approxPolyDP(cnt_orig, 0.03 * peri, True)

        if len(approx) == 4:
            pts_4 = approx.reshape(4, 2).astype("float32")
        else:
            # ถ้า approx ไม่ได้ 4 จุดพอดี ให้ใช้ minAreaRect คำนวณ 4 จุดมุม
            rect = cv2.minAreaRect(cnt_orig)
            box_pts = cv2.boxPoints(rect)
            pts_4 = np.array(box_pts, dtype="float32")

        # จัดลำดับจุด: TL, TR, BR, BL
        ordered_pts = order_points(pts_4)

        # 6. ถ้าถ่ายไกลเกินเกณฑ์ที่กำหนด ให้ส่งผลแจ้งเตือน
        if is_too_far:
            return TrayCropResult(
                success=False,
                cropped_image=None,
                corners=ordered_pts,
                tray_bbox=(bx, by, bw, bh),
                area_ratio=area_ratio,
                is_too_far=True,
                message=f"ถ่ายภาพไกลเกินไป: ถาดมีขนาดเพียง {area_ratio:.0%} ของภาพ (เกณฑ์ขั้นต่ำ {self.min_tray_ratio:.0%}) กรุณาขยับกล้องเข้าใกล้ถาดมากขึ้น",
            )

        # 7. ทำ Perspective Warp
        tl, tr, br, bl = ordered_pts
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        calc_w = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        calc_h = int(max(height_a, height_b))

        if target_size is not None:
            tw, th = target_size
            # ถ้ากล่องถาดเป็นแนวตั้ง (calc_h > calc_w) แต่ target_size เป็นแนวนอน (tw > th)
            # ให้สลับมิติเพื่อคงสัดส่วนที่ถูกต้อง ป้องกัน SIFT keypoints ผิดรูป
            if (calc_h > calc_w and tw > th) or (calc_w > calc_h and th > tw):
                dst_w, dst_h = th, tw
            else:
                dst_w, dst_h = tw, th
        else:
            dst_w, dst_h = calc_w, calc_h

        dst_pts = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ], dtype="float32")

        # ตรวจสอบความถูกต้องของ Homography Matrix
        try:
            M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
            warped = cv2.warpPerspective(
                image, M, (dst_w, dst_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
        except Exception as e:
            logger.warning(f"Perspective Transform failed: {e}, falling back to bounding box crop")
            # Fallback: ตัดตาม Bounding Box พร้อม margin 2%
            mx = int(bw * 0.02)
            my = int(bh * 0.02)
            crop_x1 = max(0, bx - mx)
            crop_y1 = max(0, by - my)
            crop_x2 = min(w_orig, bx + bw + mx)
            crop_y2 = min(h_orig, by + bh + my)
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if target_size is not None:
                warped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
            else:
                warped = cropped

        msg = f"ตรวจพบถาดและตัดขอบสำเร็จ (พื้นที่ถาด {area_ratio:.0%})"
        if area_ratio < self.optimal_tray_ratio:
            msg += f" (แนะนำ: ขยับกล้องเข้าใกล้ขึ้นอีกเล็กน้อยเพื่อให้ได้ {self.optimal_tray_ratio:.0%}+)"

        return TrayCropResult(
            success=True,
            cropped_image=warped,
            corners=ordered_pts,
            tray_bbox=(bx, by, bw, bh),
            area_ratio=area_ratio,
            is_too_far=False,
            message=msg,
        )
