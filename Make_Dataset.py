import cv2
import mediapipe as mp
import numpy as np
import keyboard

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

max_num_hands = 1
hands = mp_hands.Hands(max_num_hands=max_num_hands,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

f = open('./data/gesture.csv', 'a')
labelling = input("라벨값 입력 (0.000000 ~ 33.000000) : ")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    ## opencv는 bgr 형태이기 때문에 영상 컬러를 조정 해야 함
    video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # 웹캠에서 bgr에서 rgb로 변경
    video = cv2.flip(video, 1)

    video.flags.writeable = False                     # false로 쓰기 불가능한 상태로 설정

    # Make detection
    results = hands.process(video)                     # 감지

    video.flags.writeable = True                      # True로 쓰기 가능한 상태로 설정
    video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)    # rgb에서 bgr로 변경하여 opencv로 작동하도록 설정


    if results.multi_hand_landmarks is not None:      # 손 인식 했을 경우
        for res in results.multi_hand_landmarks:
            joint = np.zeros((21, 3))                 # 21개의 마디 부분 좌표 (x, y, z)를 joint에 저장
            for j,lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # 벡터 계산
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],:]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],:]
            v = v2 - v1

            # 벡터 길이 계산 (Normalize v)
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # arcos을 이용하여 15개의 angle 구하기
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18],:],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]))

            # radian 값을 degree로 변경
            angle = np.degrees(angle)


            ## 각도 계산한 것을 csv 파일로 저장
            if keyboard.is_pressed("s"):         ## 's' 를 누를 경우 해당 영상의 Hand 각도 값이 csv에 저장
                for ang in angle:
                    f.write(str(round(ang, 6)))
                    f.write(',')
                f.write(labelling)              ## 라벨링값 수정하면서 실행 (0.000000 ~ 33.000000)
                f.write('\n')
                print("ok")

            mp_drawing.draw_landmarks(video, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand", video)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

f.close()