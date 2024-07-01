import cv2
import numpy as np
import tensorflow as tf

# Carregar o modelo pré-treinado (MobileNet SSD treinado no COCO dataset)
model_path = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_path)

# IDs das classes que representam capacetes (exemplo fictício)
helmet_class_ids = [1]  # Verifique o ID correto para capacetes no seu modelo/dataset

# Função para fazer a detecção de objetos
def detect_objects(image):
    input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)
    detections = model(input_tensor)
    return detections

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preparar a imagem
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (300, 300))

    # Fazer a detecção de objetos
    detections = detect_objects(input_image)

    # Processar as detecções
    for i in range(int(detections['num_detections'][0])):
        score = detections['detection_scores'][0][i].numpy()
        if score < 0.5:  # Aumentar o threshold de confiança se necessário
            continue

        class_id = int(detections['detection_classes'][0][i].numpy())
        bbox = detections['detection_boxes'][0][i].numpy()

        # Verificar se a detecção é um capacete
        if class_id in helmet_class_ids:
            h, w, _ = frame.shape
            y_min, x_min, y_max, x_max = bbox
            x_min = int(x_min * w)
            x_max = int(x_max * w)
            y_min = int(y_min * h)
            y_max = int(y_max * h)

            # Desenhar a caixa delimitadora
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, 'Capacete', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar a imagem
    cv2.imshow('Detecção de Capacetes', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar as janelas
cap.release()
cv2.destroyAllWindows()
