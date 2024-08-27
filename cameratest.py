import cv2

# เปิดกล้อง
cap = cv2.VideoCapture(1)

# ตรวจสอบว่ากล้องเปิดได้หรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถจับภาพจากกล้องได้")
            break
        
        # แสดงภาพ
        cv2.imshow('Camera', frame)
        
        # ออกเมื่อกดปุ่ม 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
