import cv2
import numpy as np
import pandas as pd
import torch
import pathlib
import cvzone

# Fix lỗi PosixPath trên Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Tải mô hình YOLOv5 đã huấn luyện
model_path = "best.pt"  # Đường dẫn tới mô hình YOLOv5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sử dụng GPU nếu có
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)

# Biến toàn cục lưu tọa độ chuột
cursor_pos = (0, 0)

# Hàm callback để lấy tọa độ chuột
def mouse_move(event, x, y, flags, param):
    global cursor_pos
    if event == cv2.EVENT_MOUSEMOVE:  # Khi chuột di chuyển
        cursor_pos = (x, y)

# Hàm phát hiện và đếm người
def detect_and_count_human(frame, polygon_coords):
    # Chuyển khung hình sang định dạng PyTorch
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Lấy kết quả phát hiện

    human_count = 0
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_id = int(class_id)

        # Kiểm tra nếu đối tượng là 'person' (class_id = 0)
        if class_id == 0:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Kiểm tra nếu đối tượng nằm trong vùng hình bình hành
            if cv2.pointPolygonTest(polygon_coords, (center_x, center_y), False) >= 0:
                human_count += 1

            # Vẽ bounding box và nhãn
            cvzone.cornerRect(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), l=20, t=2, colorR=(0, 255, 0))
            cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    return frame, human_count

# Khởi tạo video capture
cap = cv2.VideoCapture("queue.mp4")  # Đường dẫn tới video

# Tọa độ hình bình hành (có thể chỉnh sửa trực tiếp trong khi chạy)
parallelogram_coords = np.array([[70, 50], [150, 25], [300, 200], [180, 250]])

# Gắn sự kiện chuột
cv2.namedWindow("Video Frame")
cv2.setMouseCallback("Video Frame", mouse_move)

while True:
    ret, frame = cap.read()

    # Nếu hết video, quay lại khung hình đầu
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Xử lý phát hiện và đếm
    frame, human_count = detect_and_count_human(frame, parallelogram_coords)

    # Vẽ hình bình hành
    cv2.polylines(frame, [parallelogram_coords], isClosed=True, color=(0, 0, 255), thickness=2)

    # Hiển thị tọa độ các đỉnh
    for point in parallelogram_coords:
        x, y = point
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Đánh dấu đỉnh
        cv2.putText(frame, f"({x}, {y})", (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Hiển thị tọa độ chuột
    cv2.putText(frame, f"Mouse Position: {cursor_pos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.circle(frame, cursor_pos, 5, (255, 0, 0), -1)  # Đánh dấu vị trí chuột

    # Hiển thị số lượng người đếm được
    cv2.putText(frame, f"Count: {human_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("Video Frame", frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
