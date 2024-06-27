import os
import cv2

# Função para criar diretórios se não existirem
def create_user_directories(user_name):
    base_path = os.path.join('usuarios', user_name)
    helmet_path = os.path.join(base_path, 'com_capacete')
    no_helmet_path = os.path.join(base_path, 'sem_capacete')
    os.makedirs(helmet_path, exist_ok=True)
    os.makedirs(no_helmet_path, exist_ok=True)
    return helmet_path, no_helmet_path

# Função para capturar e salvar imagens
def capture_images(video_capture, save_path, num_images, prompt):
    print(prompt)
    count = 0
    while count < num_images:
        ret, frame = video_capture.read()
        if not ret:
            break
        cv2.imshow('Capturing Images', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            img_name = os.path.join(save_path, f'image_{count + 1}.jpg')
            cv2.imwrite(img_name, frame)
            print(f"Imagem {count + 1} salva em: {img_name}")
            count += 1
    cv2.destroyWindow('Capturing Images')

# Solicitar o nome do usuário
user_name = input("Digite o nome do usuário: ")

# Criar diretórios para o usuário
helmet_path, no_helmet_path = create_user_directories(user_name)

# Iniciar a captura da webcam
video_capture = cv2.VideoCapture(0)

# Capturar imagens com capacete
capture_images(video_capture, helmet_path, 7, "Coloque o capacete e pressione 'c' para capturar cada imagem. Capture 7 imagens.")

# Capturar imagens sem capacete
capture_images(video_capture, no_helmet_path, 7, "Remova o capacete e pressione 'c' para capturar cada imagem. Capture 7 imagens.")

# Liberar a captura da webcam
video_capture.release()
cv2.destroyAllWindows()

print("Captura de imagens concluída. As imagens foram salvas nas respectivas pastas.")
