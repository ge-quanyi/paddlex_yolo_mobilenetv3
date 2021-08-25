import paddlex as pdx
import cv2


cap = cv2.VideoCapture(1)
cap.set(3, 800)
cap.set(4, 600)
# frame = cv2.imread("test/test.jpg")
model = pdx.load_model('output/yolov3_mobilenetv3/best_model')
while True:
    # get a frame
    ret, frame = cap.read()
    # show a frame
    result = model.predict(frame)
    goal = pdx.det.visualize(frame, result, threshold=0.3, save_dir=None)
    cv2.imshow("capture", goal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
