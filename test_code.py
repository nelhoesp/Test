import cv2
import numpy as np

PATH = "D:/Nelho/Universidad/Ciclo VIII/Proyecto Glaucoma/Dataset/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train/Glaucoma_Positive"
kernel = (11,11)

img = cv2.imread(PATH + '/072.jpg')

# Conversión a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Realce de contraste
alpha = 1.5     # Factor de escala para el contraste
beta = 1       # Valor de desplazamiento para el brillo
adjusted_img = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# Umbralización
_, umbral = cv2.threshold(adjusted_img, 200, 255, cv2.THRESH_BINARY)

# Transformaciones morfológicas
opening = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Quedarse con el área más grande
# Etiquetar los componentes conectados
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)

# Encontrar el área más grande (exceptuando el fondo)
largest_area = 0
largest_label = 1
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > largest_area:
        largest_area = area
        largest_label = i

largest_component = (labels == largest_label).astype(np.uint8) * 255

# Generacion de la mascara binaria
mask = largest_component

# Segmentación de la imagen
result = cv2.bitwise_and(img,img, mask= mask)

cv2.imshow("Original", cv2.resize(img, (int(img.shape[1]*0.2),int(img.shape[0]*0.2))))
cv2.imshow("Escala de grises", cv2.resize(gray, (int(img.shape[1]*0.2),int(img.shape[0]*0.2))))
cv2.imshow("Realce de contraste", cv2.resize(adjusted_img, (int(img.shape[1]*0.2),int(img.shape[0]*0.2))))
cv2.imshow("Umbral", cv2.resize(umbral, (int(img.shape[1]*0.2),int(img.shape[0]*0.2))))
cv2.imshow("Transformacion", cv2.resize(largest_component, (int(img.shape[1]*0.2),int(img.shape[0]*0.2))))
cv2.imshow("Segmentacion", cv2.resize(result, (int(img.shape[1]*0.2),int(img.shape[0]*0.2))))
cv2.waitKey(0)
cv2.destroyAllWindows()
