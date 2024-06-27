import cv2
import tensorflow as tf
import numpy as np

# Carregar modelo pré-treinado
model = tf.saved_model.load('dataset/capacetes')

# Função para realizar a detecção
def detect_helmet(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    detections = model(input_tensor)

    return detections

# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção no frame capturado
    detections = detect_helmet(frame)

    # Processar as detecções e desenhar caixas delimitadoras
    for detection in detections['detection_boxes']:
        ymin, xmin, ymax, xmax = detection
        (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                      ymin * frame.shape[0], ymax * frame.shape[0])
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

    # Mostrar o frame com as detecções
    cv2.imshow('Detecção de Capacete', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
