from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face

from PIL import Image
import matplotlib.pyplot as plt

img_path = '../s1_VIS_00003_003.jpg'
img = Image.open(img_path)
img = img.convert('RGB')

_ , landmark = detect_faces(img)
facial_5_points = [[landmark[0][j],landmark[0][j+5]] for j in range(5)]
print(facial_5_points)
left_eye_landmark = facial_5_points[0]
img = img.crop((left_eye_landmark[0]-11,left_eye_landmark[1]-11,left_eye_landmark[0]+11,left_eye_landmark[1]+11))
plt.imshow(img)
plt.show()

