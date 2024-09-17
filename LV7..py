#1

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix



num_classes = 10
input_shape = (28, 28, 1)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))



plt.figure(1)
plt.imshow(x_train[0,:,:],cmap='gray')
plt.show()


x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255


x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")



y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)




model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(16,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.summary()



model.compile(loss='categorical_crossentropy',optimizer='Nadam',metrics=['accuracy'])




model.fit(x_train_s, y_train_s, epochs=7, batch_size=32)


loss_and_metrics=model.evaluate(x_test_s, y_test_s, batch_size=128)
print(loss_and_metrics)





model.save('model')

#2

from keras.models import load_model
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np

filename = 'test.png'

img = mpimg.imread(filename)
img = color.rgb2gray(img)
img = resize(img, (28, 28))


plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')



model = load_model('model')



digit = np.argmax(model.predict(img), axis = -1)


print("---------------------------------------------------")
print("Slika sadrzi znamenku: ", digit)

#3

import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier


cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
font = cv2.FONT_HERSHEY_SIMPLEX





pad = 15
size_th = 32
mnist_size = 28


cp = cv2.VideoCapture(0)
kernel1 = np.ones((7, 7), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)


label = "unkown"

while True:

    ret, frame = cp.read(0)

    
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    v = np.median(gray_img)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edge_img = cv2.Canny(gray_img, lower, upper)
    img_preprocessed = cv2.dilate(edge_img, kernel1, iterations=1)
    img_preprocessed = cv2.erode(img_preprocessed, kernel2, iterations=1)

    
    _, contours, _ = cv2.findContours(img_preprocessed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = [rect for rect in rects if rect[2] >= 3 and rect[3] >= 8]

    
    for rect in rects:

        x, y, w, h = rect

       
        cropped_digit = img_preprocessed[y - pad:y + h + pad, x - pad:x + w + pad]
        cropped_digit = cropped_digit / 255.0

       
        if cropped_digit.shape[0] >= size_th and cropped_digit.shape[1] >= size_th:
            cropped_digit = cv2.resize(cropped_digit, (mnist_size, mnist_size))
        else:
            continue

        
        cv2.rectangle(frame, (x - pad, y - pad), (x + pad + w, y + pad + h), color=(255, 255, 0))

        cv2.putText(frame, label, (rect[0], rect[1]), font,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    
    cv2.imshow("frame", frame)
    cv2.imshow("edges", img_preprocessed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
