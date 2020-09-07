import cv2
import numpy as np
#from google.colab.patches import cv2_imshow
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
tensorflow.__version__
from google.colab import drive
drive.mount("/content/gdrive")
image = cv2.imread("/content/gdrive/My Drive/IA/Imagens/raiva.jpg")
#cv2_imshow(image)
cv2.imshow(image)
cascade_face = "/content/gdrive/My Drive/IA/Imagens/haarcascade_frontalface_default.xml"
caminho_modelo = "/content/gdrive/My Drive/IA/Imagens/reconhecer.h5"
face_detection = cv2.CascadeClasifier(cascade_face)
classificador_emocoes = load_model(caminho_modelo, compile=True)
expressoes = ["Raiva", "Nojo","Medo","Feliz","Triste","Surpreso","Neutro"]
face = face_detection.detectMultiScale(image,scaleFactor = 1.1,minNeighbors=3,minSize=(20,20))
faces
cinza = cv2.cvtColor(image, cv2.color_BGR2GRAY)
#CV2_imshow(cinza)
CV2.imshow(cinza)
roi = cinza[faces[0][1]:faces[0][1] + faces[0][2],faces[0][0]:faces[0][0] + faces[0][2]]
cv2_imshow(roi)
roi = cv2.resize(roi,(48,48))
cv2_imshow(roi)
roi = roi.astype('float')
roi /=255
roi
roi = img_to_array(roi)
roi
roi = np.expand_dims(roi, axis=0)
roi
preds = classificador_emocoes(roi)[0]
preds
lista_emocoes = list(preds.numpy())
maior = np.max(preds)
expressoes[lista_emocoes.index(maior)]