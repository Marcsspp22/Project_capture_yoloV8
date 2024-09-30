import cv2
from ultralytics import YOLO

# Carrega o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')  # ou 'yolov8s.pt', 'yolov8m.pt', etc.

# Captura de vídeo da webcam virtual criada pela IVCam
cap = cv2.VideoCapture(0)  # ou cap = cv2.VideoCapture(1)

# Armazena a posição anterior da caixa delimitadora
prev_box = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza a detecção de objetos no frame usando o modelo YOLO
    results = model(frame)

    # Itera sobre os resultados da detecção
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Obtém as coordenadas da caixa delimitadora e a classe detectada
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])

            # Verifica se a classe detectada é 'person' (0)
            if cls == 0:
                # Desenha a caixa delimitadora na imagem
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Verifica se houve movimento comparando com a posição anterior
                if prev_box is not None:
                    prev_x, prev_y = (prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2
                    curr_x, curr_y = (x1 + x2) // 2, (y1 + y2) // 2
                    movement = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5  # Distância Euclidiana

                    if movement > 10:  # Ajuste o limiar de movimento conforme necessário
                        cv2.putText(frame, "Andando", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Atualiza a posição anterior da caixa delimitadora
                prev_box = (x1, y1, x2, y2)

    # Exibe a imagem
    cv2.imshow('Deteccao de movimento', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()