import cv2
import torch
import numpy as np
from yolov5 import YOLOv5

# โหลดโมเดลจากไฟล์
model = YOLOv5('/Users/march/Documents/GitHub/yolov5/detect-animal/runs/train/exp/weights/best.pt')



# โหลดโมเดลจากไฟล์ weights
#model = torch.load('/Users/march/Documents/GitHub/yolov5/runs/train/exp/weights/best.pt')
#model.eval()  # ตั้งค่าโมเดลให้เป็นโหมดการทดสอบ

# เปิดกล้อง
#cap = cv2.VideoCapture(0)  # ใช้กล้องตัวแรก (0) หรือกำหนดเป็น path ของไฟล์วิดีโอ
cap = cv2.VideoCapture(1)  # ใช้กล้องตัวแรก
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")

# ชื่อของคลาสที่ตรวจจับได้
class_names = ['Rat', 'Toad', 'Lizard']  # เปลี่ยนชื่อให้ตรงกับโมเดลของคุณ

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถจับภาพจากกล้องได้")
        break

    # แปลงภาพจาก BGR เป็น RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().unsqueeze(0)  # แปลงเป็น tensor และเพิ่ม dimension แรก

    # ใช้โมเดลเพื่อทำการตรวจจับ
    #with torch.no_grad():
    #    results = model(img_tensor)
    results = model.predict(img_rgb)   
    # ประมวลผลผลลัพธ์
    # ผลลัพธ์อาจเป็นการบอกตำแหน่ง (bounding boxes) และคลาสที่ตรวจจับได้
    for result in results.xyxy[0]:  # results.xyxy[0] เป็นพิกัดของการตรวจจับ
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > 0.5:  # ตรวจสอบความมั่นใจ
            class_id = int(cls)
            class_name = class_names[class_id]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # แสดงภาพที่ตรวจจับ
    cv2.imshow('Animal Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()
