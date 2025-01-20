import mediapipe as mp
import cv2 
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
stage = None
counter = 0
def find_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radian = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])

    degree = np.abs(radian*180/np.pi)

    if degree > 180:
        degree = 360-degree
    return degree
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # recoloring the image to rgb, default is bgr in cv2
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # making detection
        results = pose.process(image)

        # recoloring back to rgb
        image.flags.writeable = True

        # extracting the landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # get coordinates
            lft_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            lft_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            lft_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y

            # calculating the angle
            angle = find_angle(lft_shoulder, lft_elbow, lft_wrist)

            # printing out the angle in the video window
            cv2.putText(image, str(angle),
                        tuple(np.multiply(lft_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 35 and stage == "down":
                stage = "up"
                counter += 1
                print(counter)

        except:
            pass

        # displaying curl counts
        cv2.rectangle(image, (0, 0), (200, 100), (225, 117, 16), -1)
        cv2.putText(image, 'REPS', (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # rendering detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
