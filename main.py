import cv2
import numpy as np
from ultralytics import YOLO
from math import atan2, degrees
from typing import Literal, List

class sopndylolisthesis_detection:
    def __init__(self, model_type: Literal["pt", "onnx"] = "pt"):
        """
        select model type for detection
        - "pt" for PyTorch model
        - "onnx" for ONNX model

        Args:
            model_type (Literal["pt", "onnx"], optional): Model type for detection. Defaults to "pt".

        Raises:
            ValueError: If an invalid model_type is provided.
        """
        
        allowed_types = {"pt", "onnx"}
        if model_type not in allowed_types:
            raise ValueError(f"Invalid model_type: '{model_type}'. Must be one of {allowed_types}")
        
        if model_type == "pt":
            self.model = YOLO('AP_YOLO11.pt')
        elif model_type == "onnx":
            self.model = YOLO('AP_YOLO11.onnx', task='obb')

    def Diagonal(self, box: List[np.ndarray]) -> float:
        """
        องศาจากเส้นทแยงมุมของด้านข้างกล่อง แล้วมาเฉลี่ยกัน 2 ข้าง

        Args:
            box (List[np.ndarray]): A list of four points representing the corners of the box.

        Returns:
            float: The angle of the diagonal in degrees.
        """

        vector1 = box[2] - box[0]  # เส้นทแยงมุมจากจุดที่ 1 -> 3
        vector2 = box[3] - box[1]  # เส้นทแยงมุมจากจุดที่ 2 -> 4

        # คำนวณมุมเฉลี่ยของทั้งสองเส้นทแยงมุม
        angle1 = degrees(atan2(vector1[1], vector1[0]))
        angle2 = degrees(atan2(vector2[1], vector2[0]))
        angle = (angle1 + angle2) / 2  # ค่าเฉลี่ยของมุมทั้งสอง

        # ให้มุมอยู่ในช่วง 0 ถึง 180 องศา
        angle = angle % 180
        return angle

    def sort_corners(self, 
                     points: List[np.ndarray]
                     ) -> List[np.ndarray]:
        """
        Sort the corners of a box in the order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        
        Args:
            points (List[np.ndarray]): A list of four points representing the corners of the box.

        Returns:
            List[np.ndarray]: Sorted list of points in the order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        """
        
        points = np.array(points)
        
        # หา Top-Left
        top_left = points[np.argmin(points.sum(axis=1))]
        
        # หา Bottom-Right
        bottom_right = points[np.argmax(points.sum(axis=1))]
        
        # หา Top-Right และ Bottom-Left
        remaining_points = [point for point in points if not np.array_equal(point, top_left) and not np.array_equal(point, bottom_right)]
        top_right = remaining_points[0] if remaining_points[0][1] < remaining_points[1][1] else remaining_points[1]
        bottom_left = remaining_points[1] if remaining_points[0][1] < remaining_points[1][1] else remaining_points[0]
        return [top_left, top_right, bottom_right, bottom_left]

    def detection(self, img):
        """
        ตรวจจับ Spondylolisthesis ในภาพ X-ray

        Args:
            img (_type_): _input image_

        Returns:
            _type_: _detection result, annotated image, left indicator, right indicator_
        """
        
        h, w = img.shape[:2]

        results = self.model(img, 
                    device=0, 
                    imgsz=640
                )
        
        left_point = []
        right_point = []
        angles = []
        # ตรวจสอบผลลัพธ์
        if results:
            for result in results:
                obb = result.obb  # ใช้ OBB ที่ได้จากโมเดล
                pointTocls = []
                allPoint = []

                # เข้าถึงพิกัด OBB ในรูปแบบ xyxyxyxy
                xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
                confidence = obb.conf.cpu().numpy()
                
                sorted_box = [self.sort_corners(b) for b in xyxyxyxy]
                angles = []

                # วาด OBB สำหรับแต่ละผลลัพธ์
                for box, con in zip(sorted_box, confidence):
                    # box จะมีพิกัด [x1, y1, x2, y2, x3, y3, x4, y4]
                    box = np.int32(box)  # แปลงเป็นจำนวนเต็ม
                    # วาด OBB บนภาพ

                    cv2.polylines(img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                    angle = self.Diagonal(box)
                    angles.append(angle)
                    allPoint.append(box)

                    # หาจุดที่ชิดซ้ายสุด (ค่าของ x น้อยสุด)
                    left_point.append(box[np.argmin(box[:, 0])])

                    # หาจุดที่ชิดขวาสุด (ค่าของ x มากสุด)
                    right_point.append(box[np.argmax(box[:, 0])])

                    # วาดจุด OBB
                    for i, point in enumerate(box):
                        if i == 1:
                            pointTocls.append([point, con])
                        cv2.circle(img, tuple(point), radius=10, color=(255, 0, 0), thickness=-1)  # จุดสีฟ้า

        sorted_arrays = sorted(pointTocls, key=lambda x: x[0][1])
        for point, class_name, ang in zip(sorted_arrays, ['L1', 'L2', 'L3', 'L4', 'L5'], angles):
            cv2.putText(img, f'{class_name}, {point[1]:.2f}, {ang:.2f}', (point[0][0] + 10, point[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)  # สีขาว

        left = 0
        right = 0
        if angles:
            difference = round(max(angles) - min(angles), 2)
        else:
            out = 'no detection'
            return out, None, None, None

        if difference >= 11.58:
            out = 'Spondylolisthesis'
            cv2.putText(img, 'Spondylolisthesis', (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)  # สีขาว

            left_point = np.array(left_point)
            right_point = np.array(right_point)
            leftmost_point = left_point[np.argmin(left_point[:, 0])]
            rightmost_point = right_point[np.argmax(right_point[:, 0])]

            cv2.circle(img, tuple(leftmost_point), radius=15, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, f'Dist {leftmost_point[0]}', (leftmost_point[0]-180, leftmost_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(img, tuple(rightmost_point), radius=15, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, f'Dist {w - rightmost_point[0]}', (rightmost_point[0]+40, rightmost_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if leftmost_point[0] < (w - rightmost_point[0]):
                cv2.putText(img, 'Right Laterolisthesis', (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)  # สีขาว
                right = 1
                left = 0
            else:
                cv2.putText(img, 'Left Laterolisthesis', (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
                right = 0
                left = 1

        elif difference < 11.58: #11.65
            out = 'Non Spondylolisthesis'
            cv2.putText(img, 'Non Spondylolisthesis', (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)

        return out, img, left, right
                  
if __name__ == "__main__":

    img_path = 'AP/0223-F-063Y0.jpg'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    spondy = sopndylolisthesis_detection(model_type="onnx")
    out, image, l, r = spondy.detection(img)
    if image is not None:
        h, w = image.shape[:2]
        image = cv2.resize(image, (h//2, w//2))
        cv2.imshow('result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()