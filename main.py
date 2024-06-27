import os
import cv2
import face_recognition as fr


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

while True:
    # Capturar frame da webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Redimensionar o frame para acelerar o processo de reconhecimento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Encontrar todas as localizações de rosto e codificações no frame atual
    face_locations = fr.face_locations(rgb_small_frame)
    face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparar o rosto capturado com os rostos conhecidos
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"
        status = ""

        # Se houver uma correspondência, usar o primeiro rosto conhecido correspondente
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            status = helmet_status[best_match_index]

        # Ajustar as coordenadas do rosto para o tamanho original do frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Colocar o nome e o status abaixo do rosto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({status})", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mostrar o frame
    cv2.imshow('Video', frame)

    # Parar a captura quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura da webcam e fechar a janela
video_capture.release()
cv2.destroyAllWindows()