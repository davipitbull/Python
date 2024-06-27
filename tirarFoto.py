import cv2

# Inicializa a captura de vídeo da webcam (0 é o índice da câmera padrão)
cap = cv2.VideoCapture(0)

# Verifica se a captura foi iniciada com sucesso
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# Contador para nomear as fotos salvas
photo_counter = 0

while True:
    # Lê um quadro da câmera
    ret, frame = cap.read()

    # Verifica se a leitura foi bem-sucedida
    if not ret:
        print("Erro: Não foi possível capturar a imagem.")
        break

    # Exibe o quadro na janela
    cv2.imshow('Câmera - Pressione "s" para salvar a foto', frame)

    # Aguarda a tecla 'q' para sair ou 's' para salvar a foto
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Salva a foto capturada
        photo_name = f'foto_{photo_counter}.jpg'
        cv2.imwrite(photo_name, frame)
        print(f'Foto salva: {photo_name}')
        photo_counter += 1

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
