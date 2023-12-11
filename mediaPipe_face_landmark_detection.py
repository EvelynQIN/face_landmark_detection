import cv2 
import mediapipe as mp 

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh 
face_mesh = mp_face_mesh.FaceMesh()

# image
image = cv2.imread("image.png")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# Facial landmarks
result = face_mesh.process(rgb_image)

for facial_landmarks in result.multi_face_landmarks:
    print(f'num of lk = {len(facial_landmarks.landmark)}')
    for i in range(468):
        pt1 = facial_landmarks.landmark[i]
        x, y, z = int(pt1.x * w), int(pt1.y * h), int(pt1.z)
        cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
        # cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))

cv2.imshow("Image", image)
cv2.waitKey(0)