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
    corners: Optional[np.ndarray] = None          # 4 จุดมุม [TL, TR, BR, BL] ในพิกัดภาพเดิม
    tray_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) ในภาพเดิม
    area_ratio: float = 1.0                      # สัดส่วนพื้นที่ถาดเทียบกับภาพทั้งหมด (0.0 - 1.0)
    is_too_far: bool = False                       # True ถ้าถ่ายไกลเกินไป (area_ratio < min_tray_ratio)
    message: str = ""

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

    @staticmethod
    def is_red_chest_enclosure(image: np.ndarray) -> bool:
        """ตรวจสอบว่าภาพถ่ายมีขอบเขตตู้หรือลิ้นชักเครื่องมือสีแดงล้อมรอบหรือไม่"""
        if image is None or image.size == 0:
            return False
        h, w = image.shape[:2]
        total = float(h * w)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([16, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([162, 45, 35]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(m1, m2)

        red_ratio = np.sum(red_mask > 0) / total
        top_r = np.sum(red_mask[:int(0.12 * h), :] > 0) / (0.12 * total)
        bot_r = np.sum(red_mask[int(0.88 * h):, :] > 0) / (0.12 * total)
        left_r = np.sum(red_mask[:, :int(0.12 * w)] > 0) / (0.12 * total)
        right_r = np.sum(red_mask[:, int(0.88 * w):] > 0) / (0.12 * total)
        border_sum = top_r + bot_r + left_r + right_r
        return (red_ratio >= 0.12 and border_sum >= 0.45) or (red_ratio >= 0.22)

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

        # 0. ตรวจสอบแนวภาพ: หากภาพเป็นแนวตั้ง (Height > Width) ปรับหมุนเป็นแนวนอน (Landscape) อัตโนมัติ
        was_rotated = False
        if image.shape[0] > image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            was_rotated = True
            logger.info("TrayBoxCropper: ตรวจพบภาพแนวตั้ง -> ปรับหมุนเป็นแนวนอน 90° อัตโนมัติ")

        h_orig, w_orig = image.shape[:2]
        total_img_area = float(h_orig * w_orig)

        # 1. การตรวจจับถาดโฟมดำในตู้/ลิ้นชักสีแดง (Red Chest / Cabinet & Black Foam Isolation)
        # ตัดขอบตู้เหล็กสีแดง ก้นลิ้นชัก และวัตถุแปลกปลอมในรางข้าง (เช่น ปืนลม DG-10, เหล็กงัด) ออกโดยอัตโนมัติ
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([16, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([162, 45, 35]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(m1, m2)

        red_ratio = np.sum(red_mask > 0) / total_img_area
        top_r = np.sum(red_mask[:int(0.12 * h_orig), :] > 0) / (0.12 * total_img_area)
        bot_r = np.sum(red_mask[int(0.88 * h_orig):, :] > 0) / (0.12 * total_img_area)
        left_r = np.sum(red_mask[:, :int(0.12 * w_orig)] > 0) / (0.12 * total_img_area)
        right_r = np.sum(red_mask[:, int(0.88 * w_orig):] > 0) / (0.12 * total_img_area)
        border_sum = top_r + bot_r + left_r + right_r

        is_red_chest = (red_ratio >= 0.12 and border_sum >= 0.45) or (red_ratio >= 0.22)

        if is_red_chest:
            k_size = max(15, int(min(h_orig, w_orig) * 0.03))
            if k_size % 2 == 0:
                k_size += 1
            k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close, iterations=2)

            cnts, hier = cv2.findContours(red_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hier is not None and len(hier) > 0:
                hier = hier[0]
                holes = []
                for i, c in enumerate(cnts):
                    if hier[i][3] != -1:  # โพรงด้านในกรอบสีแดง (Inner cavity inside red frame)
                        area = cv2.contourArea(c)
                        if area >= 0.10 * total_img_area:
                            bx, by, bw, bh = cv2.boundingRect(c)
                            cx, cy = bx + bw / 2.0, by + bh / 2.0
                            dist = np.hypot(cx - w_orig / 2.0, cy - h_orig / 2.0) / np.hypot(w_orig / 2.0, h_orig / 2.0)
                            score = area * (1.0 - 0.4 * dist)
                            holes.append((score, bx, by, bw, bh))
                if holes:
                    holes.sort(key=lambda x: x[0], reverse=True)
                    _, hx, hy, hw, hh = holes[0]

                    # กำจัดรางสีแดงด้านข้าง (เช่น ปืนลม DG-10 ด้านซ้าย หรือเหล็กงัดด้านขวา)
                    hole_hsv = hsv[hy:hy + hh, hx:hx + hw]
                    hm1 = cv2.inRange(hole_hsv, np.array([0, 45, 35]), np.array([16, 255, 255]))
                    hm2 = cv2.inRange(hole_hsv, np.array([162, 45, 35]), np.array([180, 255, 255]))
                    h_red = cv2.bitwise_or(hm1, hm2)

                    col_r = np.sum(h_red > 0, axis=0) / float(hh)
                    row_r = np.sum(h_red > 0, axis=1) / float(hw)

                    valid_cols = np.where(col_r <= 0.22)[0]
                    valid_rows = np.where(row_r <= 0.25)[0]

                    if len(valid_cols) > 0 and len(valid_rows) > 0:
                        pad_x = int(hw * 0.035)  # 3.5% safe margin
                        pad_y = int(hh * 0.035)
                        fx1 = max(0, int(valid_cols[0]) - pad_x)
                        fx2 = min(hw, int(valid_cols[-1]) + pad_x)
                        fy1 = max(0, int(valid_rows[0]) - pad_y)
                        fy2 = min(hh, int(valid_rows[-1]) + pad_y)

                        final_x = int(hx + fx1)
                        final_y = int(hy + fy1)
                        final_w = int(fx2 - fx1)
                        final_h = int(fy2 - fy1)
                    else:
                        final_x, final_y, final_w, final_h = int(hx), int(hy), int(hw), int(hh)

                    tray_area = float(final_w * final_h)
                    area_ratio = tray_area / total_img_area
                    is_too_far = area_ratio < self.min_tray_ratio

                    ordered_pts = np.array([
                        [final_x, final_y],
                        [final_x + final_w, final_y],
                        [final_x + final_w, final_y + final_h],
                        [final_x, final_y + final_h]
                    ], dtype="float32")

                    if is_too_far:
                        return TrayCropResult(
                            success=False,
                            cropped_image=None,
                            corners=ordered_pts,
                            tray_bbox=(final_x, final_y, final_w, final_h),
                            area_ratio=area_ratio,
                            is_too_far=True,
                            message=f"ถ่ายภาพไกลเกินไป: ถาดมีขนาดเพียง {area_ratio:.0%} ของภาพ (เกณฑ์ขั้นต่ำ {self.min_tray_ratio:.0%}) กรุณาขยับกล้องเข้าใกล้ถาดมากขึ้น",
                        )

                    crop_img = image[final_y:final_y + final_h, final_x:final_x + final_w]
                    if target_size is not None:
                        tw, th = target_size
                        if tw < th:
                            tw, th = th, tw
                        crop_img = cv2.resize(crop_img, (tw, th), interpolation=cv2.INTER_LANCZOS4)

                    if crop_img.shape[0] > crop_img.shape[1]:
                        crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)

                    rot_note = " [ปรับเป็นแนวนอนอัตโนมัติ]" if was_rotated else ""
                    logger.info(f"TrayBoxCropper: ตรวจพบถาดในตู้/ลิ้นชักสีแดง -> ตัดขอบแม่นยำ x={final_x}, y={final_y}, w={final_w}, h={final_h} (พื้นที่ {area_ratio:.0%})")
                    return TrayCropResult(
                        success=True,
                        cropped_image=crop_img,
                        corners=ordered_pts,
                        tray_bbox=(final_x, final_y, final_w, final_h),
                        area_ratio=area_ratio,
                        is_too_far=False,
                        message=f"ตรวจจับถาดโฟมดำและตัดขอบตู้/ลิ้นชักสีแดงสำเร็จ (พื้นที่ถาด {area_ratio:.0%}){rot_note}",
                    )

        # 2. ย่อภาพเพื่อหา Contour และขอบสี่เหลี่ยมได้อย่างรวดเร็วและเสถียร (กรณีถาดตั้งบนโต๊ะ/พื้นหลังทั่วไป)
        scale = 600.0 / max(h_orig, w_orig)
        small = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        small_area = float(sh * sw)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. ค้นหาขอบเขตสี่เหลี่ยมของถาดเทียบกับพื้นหลัง (Background Border & Corner Detection)
        candidate_quads = []

        # 2.1 Multi-threshold Canny Edges เพื่อจับเส้นขอบรอบนอกของถาดกับพื้นหลัง
        for low_t, high_t in [(30, 100), (50, 150), (20, 80)]:
            edges = cv2.Canny(blurred, low_t, high_t)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if 0.15 * small_area <= area <= 0.98 * small_area:
                    hull = cv2.convexHull(c)
                    peri = cv2.arcLength(hull, True)
                    found_4 = False
                    for eps in [0.02, 0.03, 0.04, 0.05]:
                        approx = cv2.approxPolyDP(hull, eps * peri, True)
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            pts = approx.reshape(4, 2) / scale
                            bx_t, by_t, bw_t, bh_t = cv2.boundingRect(hull)
                            solidity = area / float(max(1, bw_t * bh_t))
                            candidate_quads.append((area * 1.2 * solidity, pts, c))
                            found_4 = True
                            break
                    if not found_4:
                        rect = cv2.minAreaRect(hull)
                        box = cv2.boxPoints(rect) / scale
                        candidate_quads.append((area * 0.95, box, c))

        # 2.2 Otsu & Inverse Otsu Thresholding เพื่อแยกพื้นหลังออกจากตัวถาด
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for mask in [otsu, cv2.bitwise_not(otsu)]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if 0.15 * small_area <= area <= 0.98 * small_area:
                    hull = cv2.convexHull(c)
                    peri = cv2.arcLength(hull, True)
                    found_4 = False
                    for eps in [0.02, 0.03, 0.04, 0.05]:
                        approx = cv2.approxPolyDP(hull, eps * peri, True)
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            pts = approx.reshape(4, 2) / scale
                            bx_t, by_t, bw_t, bh_t = cv2.boundingRect(hull)
                            solidity = area / float(max(1, bw_t * bh_t))
                            candidate_quads.append((area * 1.15 * solidity, pts, c))
                            found_4 = True
                            break
                    if not found_4:
                        rect = cv2.minAreaRect(hull)
                        box = cv2.boxPoints(rect) / scale
                        candidate_quads.append((area * 0.90, box, c))

        # 3. เลือกสี่เหลี่ยมมุมถาดที่ดีที่สุด
        if not candidate_quads:
            out_img = image
            if target_size is not None:
                tw, th = target_size
                if tw < th:
                    tw, th = th, tw
                out_img = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            return TrayCropResult(
                success=True,
                cropped_image=out_img,
                corners=np.array([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]], dtype="float32"),
                tray_bbox=(0, 0, w_orig, h_orig),
                area_ratio=1.0,
                is_too_far=False,
                message="ไม่พบขอบพื้นหลังรอบนอก (ภาพถูกถ่ายครอบคลุมเต็มเฟรมอยู่แล้ว)" + (" [ปรับเป็นแนวนอนอัตโนมัติ]" if was_rotated else ""),
            )

        candidate_quads.sort(key=lambda x: x[0], reverse=True)
        best_score, best_pts, best_cnt = candidate_quads[0]

        # สเกล contour และคำนวณพื้นที่
        cnt_orig = (best_cnt / scale).astype(np.int32)
        bx, by, bw, bh = cv2.boundingRect(cnt_orig)
        tray_area = float(cv2.contourArea(cnt_orig))
        area_ratio = tray_area / total_img_area

        # จัดลำดับ 4 มุม: TL, TR, BR, BL
        ordered_pts = order_points(np.array(best_pts, dtype="float32"))

        # ตรวจสอบระยะถ่ายภาพ
        is_too_far = area_ratio < self.min_tray_ratio

        # หากถาดกินพื้นที่ >= 75% ของภาพอยู่แล้ว ถือว่าระยะดีเยี่ยม ไม่ต้องตัดขอบ
        if area_ratio >= 0.75:
            return TrayCropResult(
                success=True,
                cropped_image=image,
                corners=ordered_pts,
                tray_bbox=(bx, by, bw, bh),
                area_ratio=area_ratio,
                is_too_far=False,
                message=f"ภาพถ่ายเต็มเฟรมดีเยี่ยม (ถาดกินพื้นที่ {area_ratio:.0%} ของภาพ) — ใช้งานภาพคมชัดสูงสุด" + (" [ปรับเป็นแนวนอนอัตโนมัติ]" if was_rotated else ""),
            )

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

        # 4. Safe Padding: ขยายขอบปลอดภัย 6% ออกไปรอบทิศทาง เพื่อป้องกันไม่ให้ตัดโดนเครื่องมือริมขอบ
        centroid = np.mean(ordered_pts, axis=0)
        expand_ratio = 1.06
        expanded_pts = centroid + (ordered_pts - centroid) * expand_ratio
        expanded_pts[:, 0] = np.clip(expanded_pts[:, 0], 0, w_orig - 1)
        expanded_pts[:, 1] = np.clip(expanded_pts[:, 1], 0, h_orig - 1)
        ordered_pts = expanded_pts

        # 5. ทำ Perspective Warp
        tl, tr, br, bl = ordered_pts
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        calc_w = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        calc_h = int(max(height_a, height_b))

        # รับประกันว่ามิติผลลัพธ์เป็นแนวนอนเสมอ (Landscape: width >= height)
        if calc_w < calc_h:
            calc_w, calc_h = calc_h, calc_w

        if target_size is not None:
            tw, th = target_size
            if tw < th:
                tw, th = th, tw
            dst_w, dst_h = tw, th
        else:
            dst_w, dst_h = calc_w, calc_h

        dst_pts = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ], dtype="float32")

        try:
            M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
            warped = cv2.warpPerspective(
                image, M, (dst_w, dst_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
        except Exception as e:
            logger.warning(f"Perspective Transform failed: {e}, falling back to bounding box crop")
            mx = int(bw * 0.03)
            my = int(bh * 0.03)
            crop_x1 = max(0, bx - mx)
            crop_y1 = max(0, by - my)
            crop_x2 = min(w_orig, bx + bw + mx)
            crop_y2 = min(h_orig, by + bh + my)
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if target_size is not None:
                tw, th = target_size
                if tw < th:
                    tw, th = th, tw
                warped = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            else:
                warped = cropped

        # รับประกันผลลัพธ์สุดท้ายเป็นแนวนอนเสมอ
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        rot_note = " [ปรับเป็นแนวนอนอัตโนมัติ]" if was_rotated else ""
        msg = f"จับมุมขอบสี่เหลี่ยมถาดและตัดขอบสำเร็จ (พื้นที่ถาด {area_ratio:.0%}){rot_note}"
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
