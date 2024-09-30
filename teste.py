import cv2
from ultralytics import YOLO

person_model = YOLO('yolov8n.pt')  
hand_model = YOLO('runs/train/yolov8n_hands_detection3/weights/best.pt') 

cap = cv2.VideoCapture(0)  

prev_person_box = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    person_results = person_model(frame)

    for r in person_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])

            if cls == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Pessoa", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if prev_person_box is not None:
                    prev_x, prev_y = (prev_person_box[0] + prev_person_box[2]) // 2, (prev_person_box[1] + prev_person_box[3]) // 2
                    curr_x, curr_y = (x1 + x2) // 2, (y1 + y2) // 2
                    movement = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5 

                    if movement > 10: 
                        cv2.putText(frame, "Andando", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                prev_person_box = (x1, y1, x2, y2)

    hand_results = hand_model(frame)
    confidence_threshold = 0.7  


    for r in hand_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])

            if cls in [0, 1]:  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Mao", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Deteccao de Pessoa e Mao', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
