import cv2
import mediapipe as mp
import math


def angle_finder(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


cap = cv2.VideoCapture(0)
#writer = cv2.VideoWriter("video_kaydı.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 20, (int(cap.get(3)), int(cap.get(4))))
mp_pose = mp.solutions.pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
    while cap.isOpened():
        success, image = cap.read()
        h, w, _ = image.shape
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:  # koordinatlar

            x8 = (results.pose_landmarks.landmark[8].x) * w  # right ear, x, for left x7
            y8 = (results.pose_landmarks.landmark[8].y) * h  # right ear, y, for left y7

            x12 = (results.pose_landmarks.landmark[12].x) * w  # right shoulder, x, for left x11
            y12 = (results.pose_landmarks.landmark[12].y) * h  # right shoulder, y, for left y11

            x24 = (results.pose_landmarks.landmark[24].x) * w  # right belly, x, for left x23
            y24 = (results.pose_landmarks.landmark[24].y) * h  # right belly, y, for left y23

            x26 = (results.pose_landmarks.landmark[26].x) * w  # right knee, x, for left x25
            y26 = (results.pose_landmarks.landmark[26].y) * h  # right knee, y, for left y25

            cv2.circle(image, (int(x12), int(y12)), 10, (0, 255, 0), -1)
            cv2.circle(image, (int(x24), int(y24)), 10, (0, 255, 0), -1)

            cv2.line(image, (int(x24), int(y24)), (int(x12), int(y12)), (0, 255, 0), 2)
            cv2.line(image, (int(x24), int(y24)), (int(x26), int(y26)), (0, 255, 0), 2)
            cv2.line(image, (int(x12), int(y12)), (int(x8 / 1.2), int(y8)), (0, 255, 0), 2)

            angle1 = angle_finder((x12, y12), (x24, y24), (x26, y26))
            cv2.putText(image, f'back angle : {int(angle1)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

            if angle1 < 100:
                cv2.circle(image, (int(x24), int(y24)), 10, (0, 0, 255), -1)
                cv2.line(image, (int(x24), int(y24)), (int(x12), int(y12)), (0, 0, 255), 2)
                cv2.line(image, (int(x24), int(y24)), (int(x26), int(y26)), (0, 0, 255), 2)

            angle2 = angle_finder((x8 / 1.2, y8), (x12, y12), (x24, y24))
            cv2.putText(image, f'neck angle : {int(angle2)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

            if ((angle1 > 115) & (angle2 < 142) | (115 > angle1 > 100) & (angle2 < 150) | (100 > angle1) & (angle2 < 162)):
                cv2.circle(image, (int(x12), int(y12)), 10, (0, 0, 255), -1)
                cv2.line(image, (int(x12), int(y12)), (int(x8 / 1.2), int(y8)), (0, 0, 255), 2)
                cv2.line(image, (int(x12), int(y12)), (int(x24), int(y24)), (0, 0, 255), 2)

        cv2.imshow('Pose', image)
        #writer.write(image)  # save
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
#writer.release()
cv2.destroyAllWindows()

