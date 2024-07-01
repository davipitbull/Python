import os
import cv2
import face_recognition as fr
import time
from datetime import datetime

# Função para carregar imagens de uma pasta e codificá-las
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = fr.load_image_file(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = fr.face_encodings(img)
        if face_encodings:
            images.append(face_encodings[0])
    return images

# Diretório principal onde estão os usuários
user_directory = 'usuarios'

# Listas para armazenar os nomes, codificações e status de capacete
known_face_names = []
known_face_encodings = []
helmet_status = []

# Percorrer cada usuário no diretório
for user in os.listdir(user_directory):
    user_path = os.path.join(user_directory, user)
    if os.path.isdir(user_path):
        # Carregar imagens com capacete
        helmet_images = load_images_from_folder(os.path.join(user_path, 'com_capacete'))
        known_face_encodings.extend(helmet_images)
        known_face_names.extend([user] * len(helmet_images))
        helmet_status.extend(['com capacete'] * len(helmet_images))

        # Carregar imagens sem capacete
        no_helmet_images = load_images_from_folder(os.path.join(user_path, 'sem_capacete'))
        known_face_encodings.extend(no_helmet_images)
        known_face_names.extend([user] * len(no_helmet_images))
        helmet_status.extend(['sem capacete'] * len(no_helmet_images))

# Iniciar a captura da webcam
video_capture = cv2.VideoCapture(0)
identified_time = {}
photo_directory = 'capturas'

# Certifique-se de que o diretório de capturas exista
if not os.path.exists(photo_directory):
    os.makedirs(photo_directory)

# Ajustar o fator de redimensionamento (menor valor = maior raio de identificação)
resize_factor = 0.5  # Aumente para diminuir o raio de identificação, diminua para aumentar

while True:
    # Capturar frame da webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Redimensionar o frame para acelerar o processo de reconhecimento
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Encontrar todas as localizações de rosto e codificações no frame atual
    face_locations = fr.face_locations(rgb_small_frame, model='hog')  # Altere para 'cnn' para mais precisão
    face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparar o rosto capturado com os rostos conhecidos
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"
        status = ""
        remaining_time = 3

        # Se houver uma correspondência, usar o primeiro rosto conhecido correspondente
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            status = helmet_status[best_match_index]

            # Se a pessoa for identificada, iniciar ou atualizar o temporizador
            if name in identified_time:
                elapsed_time = time.time() - identified_time[name]
                remaining_time = 3 - int(elapsed_time)
                if elapsed_time >= 3:
                    # Salvar a imagem após 3 segundos
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    photo_filename = f"{photo_directory}/{name}_{status}_{current_time}.jpg"
                    cv2.imwrite(photo_filename, frame)
                    print(f"Foto salva: {photo_filename}")
                    del identified_time[name]
            else:
                identified_time[name] = time.time()
        else:
            if name in identified_time:
                del identified_time[name]

        # Ajustar as coordenadas do rosto para o tamanho original do frame
        top = int(top / resize_factor)
        right = int(right / resize_factor)
        bottom = int(bottom / resize_factor)
        left = int(left / resize_factor)

        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Colocar o nome e o status dentro da caixinha abaixo do rosto
        text = f"{name} ({status})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1)
        cv2.rectangle(frame, (left, bottom), (right, bottom + text_height + 20), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, text, (left + 6, bottom + text_height + 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Desenhar o temporizador na tela
        if name in identified_time:
            timer_text = f"Captura em: {remaining_time} s"
            (timer_text_width, timer_text_height), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1)
            cv2.rectangle(frame, (left, top - timer_text_height - 20), (right, top), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, timer_text, (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Mostrar o frame
    cv2.imshow('Video', frame)

    # Parar a captura quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura da webcam e fechar a janela
video_capture.release()
cv2.destroyAllWindows()