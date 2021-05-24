import numpy as np
from keras.models import model_from_json
import operator
import cv2

# Read the JSON after CNN_train
json_file = open("./model/model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("./model/model-bw.h5")

cap = cv2.VideoCapture(0)
print("Start Hand Detection")

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    roi = frame[y1:y2, x1:x2]
    cv2.imshow("Frame", frame)

    image_HSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Bounds for the skin colour
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    image_skin = cv2.inRange(image_HSV, lower, upper)
    my_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # Eliminate the noise
    image_skin = cv2.erode(image_skin, my_kernel, iterations=2)
    image_skin = cv2.dilate(image_skin, my_kernel, iterations=2)
    image_skin = cv2.GaussianBlur(image_skin, (3, 3), 0)
    image_skin = cv2.threshold(image_skin, 200, 255, cv2.THRESH_BINARY)[1]
    roi = cv2.resize(image_skin, (64, 64))
    test_image = roi
    cv2.imshow("test", test_image)

    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    prediction = {'ZERO': result[0][0],
                  'ONE': result[0][1],
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5]}

    # Sorted to see where is the max match
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.imshow("Frame", frame)

    # Exit from ESC
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
