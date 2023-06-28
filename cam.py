import cv2
import time

# 웹캠 열기
cap = cv2.VideoCapture(4)  # 0은 기본 웹캠을 의미합니다. 만약 여러 개의 웹캠이 연결되어 있다면 숫자를 변경하여 선택할 수 있습니다.

#웹캠 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # 웹캠으로부터 프레임 읽기
    now= time.time()
    ret, frame = cap.read()

    # 프레임 표시하기
    cv2.imshow('Webcam', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print("rt:",time.time()-now)

# 리소스 해제
cap.release()
cv2.destroyAllWindows()