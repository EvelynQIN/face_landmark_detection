import cv2 
import mediapipe as mp 

def online_show_video_detection(video_path, refine=False):
    # Face Mesh
    mp_face_mesh = mp.solutions.face_mesh 
    
    # Whether to further refine the landmark coordinates
    # around the eyes and lips, and output additional landmarks around the
    # irises. Default to False.
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=refine)  

    # load video 
    cap = cv2.VideoCapture(video_path)
    frames = 0

    # loop over each frame
    while True:
        
        # get the signal frame image
        ret, image = cap.read()
        if ret is False:    # end of video
            break
        frames += 1
        print(frames)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Facial landmarks
        result = face_mesh.process(rgb_image)

        for facial_landmarks in result.multi_face_landmarks:
            # print(f'num of lk = {len(facial_landmarks.landmark)}') # 468
            for i in range(468):
                pt1 = facial_landmarks.landmark[i]
                x, y, z = int(pt1.x * w), int(pt1.y * h), int(pt1.z)
                cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
                # cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))
        cv2.imshow("Image", image)
        cv2.waitKey(1)
    

def get_detected_images(video_path, refine):
    
    # Face Mesh
    mp_face_mesh = mp.solutions.face_mesh 
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=refine)

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
        h, w, _ = image.shape

        # Facial landmarks
        result = face_mesh.process(rgb_image)

        for facial_landmarks in result.multi_face_landmarks:
            print(f'num of lk = {len(facial_landmarks.landmark)}') # 468
            for i in range(len(facial_landmarks.landmark)):
                pt1 = facial_landmarks.landmark[i]
                x, y, z = int(pt1.x * w), int(pt1.y * h), int(pt1.z)
                cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
                # cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))
                
        img_arr.append(image) 
    
    cap.release()
    
    return img_arr 


def video_to_video_detection(from_path, to_path, refine):
    img_arr = get_detected_images(from_path, refine)
    h, w, _ = img_arr[0].shape
    print(f'there are {len(img_arr)} frames for video: {from_path}')
    
    out = cv2.VideoWriter(to_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for i in range(len(img_arr)):
        out.write(img_arr[i])
    out.release()








if __name__ == "__main__":
    refine = True
    from_video_path = 'detection_results_vis/video_right_side_400059.mp4'
    to_video_path = 'detection_results_vis/annotated_video_478_right_side_400059.mp4'
    video_to_video_detection(from_video_path, to_video_path, refine)
    

