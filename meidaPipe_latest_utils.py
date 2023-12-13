# **************************************************************************************************************************************#
# using the latest model (478 landmarks, more robust to extreme side views)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 
import cv2 
import numpy as np


MODEL_PATH = 'face_landmarker_v2_with_blendshapes.task' 
IMG_PATH = 'datasets/multiface/minidataset/m--20180227--0000--6795937--GHS/images/E057_Cheeks_Puffed/400059/021909.png'

def detect_one_image_one_line(img_path):

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces = 1,
        output_facial_transformation_matrixes=True)

    with FaceLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(img_path)

        # # Load the input image from a numpy array.
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        
        # Perform face landmarking on the provided single image.
        # The face landmarker must be created with the image mode.
        result = landmarker.detect(mp_image) 
        
        print("number of keypoints: ", len(result.face_landmarks[0])) 
        
        # image
        image = cv2.imread(img_path)
        h, w, _ = image.shape

        for facial_landmarks in result.face_landmarks:
            for pt in facial_landmarks:
                x, y, z = int(pt.x * w), int(pt.y * h), int(pt.z)
                cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
                # cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))

        cv2.imshow("Image", image)
        cv2.waitKey(0)

def get_detection_results_from_one_image(img_path, running_mode):
    
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_faces = 1,
        output_facial_transformation_matrixes=True)

    with FaceLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(img_path)

        # # Load the input image from a numpy array.
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        
        # Perform face landmarking on the provided single image.
        # The face landmarker must be created with the image mode.
        result = landmarker.detect(mp_image) 
        
        # image
        keypoints = []

        for pt in result.face_landmarks[0]:
            points_xyz = np.array([pt.x, pt.y, pt.z])
            keypoints.append(points_xyz)
        keypoints = np.vstack(keypoints)
        print(keypoints.shape) 
        return keypoints 