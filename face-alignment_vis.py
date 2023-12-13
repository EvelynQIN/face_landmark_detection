import cv2
import torch
import face_alignment
from skimage import io, img_as_ubyte


def detect_online_one_image(img_path):
    # face detection model
    face_detector_kwargs = {
        'back_model': True
    }
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=False,
        face_detector='blazeface',    # support detectors ['dlib', 'blazeface', 'cfd]
        face_detector_kwargs = face_detector_kwargs,
        dtype=torch.bfloat16, device='cuda',
    )

    # image
    image = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Facial landmarks
    result = fa.get_landmarks_from_image(rgb_img)

    for x, y in result[0]:
        cv2.circle(image, (int(x), int(y)), 3, (255, 255, 255), -1)
        # cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))

    cv2.imshow("Image", image)
    cv2.waitKey(0)

def get_detected_images(video_path):
    
    # Face Mesh
    face_detector_kwargs = {
        'back_model': True
    }
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, 
                                         flip_input=False,
                                         face_detector='blazeface', 
                                         dtype=torch.bfloat16, device='cuda',
                                         face_detector_kwargs = face_detector_kwargs,
                                        )

    # load video 
    cap = cv2.VideoCapture(video_path)
    
    # the annoted cv2 img array list
    img_arr = []
    
    # loop over each frame
    while True:
        
        # get the signal frame image
        ret, image = cap.read()
        if ret is False:    # end of video
            break

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Facial landmarks
        result = fa_3d.get_landmarks_from_image(rgb_image)

        for x, y, z in result[0]:
            cv2.circle(image, (int(x), int(y)), 5, (255, 255, 255), -1)
            # cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))
                
        img_arr.append(image) 
    
    cap.release()
    
    return img_arr 

def video_to_video_detection(from_path, to_path):
    img_arr = get_detected_images(from_path)
    h, w, _ = img_arr[0].shape
    print(f'there are {len(img_arr)} frames for video: {from_path}')
    
    out = cv2.VideoWriter(to_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for i in range(len(img_arr)):
        out.write(img_arr[i])
    out.release()

if __name__ == "__main__":
    from_video_path = 'detection_results_vis/video_right_side_400059.mp4'
    to_video_path = 'detection_results_vis/annotated_video_68_right_side_400059.mp4'
    video_to_video_detection(from_video_path, to_video_path)
    
    # img_path = "datasets/multiface/minidataset/m--20180227--0000--6795937--GHS/images/E057_Cheeks_Puffed/400059/021897.png"
    # # img_path = "image.png"
    # detect_online_one_image(img_path)