from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build our CNN for recognizing hand gesture
my_classifier = Sequential()

my_classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
my_classifier.add(MaxPooling2D(pool_size=(2, 2)))
my_classifier.add(Convolution2D(32, (3, 3), activation='relu'))
my_classifier.add(MaxPooling2D(pool_size=(2, 2)))
my_classifier.add(Flatten())
my_classifier.add(Dense(units=128, activation='relu'))
my_classifier.add(Dense(units=6, activation='softmax'))
my_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train our data
train_data_hand = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_data_hand = ImageDataGenerator(rescale=1. / 255)

training_set_hand = train_data_hand.flow_from_directory('data/train', target_size=(64, 64), batch_size=5,
                                                        color_mode='grayscale',
                                                        class_mode='categorical')

test_set_hand = test_data_hand.flow_from_directory('data/test', target_size=(64, 64), batch_size=5,
                                                        color_mode='grayscale',
                                                        class_mode='categorical')

# Fit the train data
my_classifier.fit(training_set_hand, steps_per_epoch=9008, epochs=4, validation_data=test_set_hand,
                  validation_steps=28)

# Save model as JSON
model_json = my_classifier.to_json()
with open("./model/model-bw.json", "w") as json_file:
    json_file.write(model_json)
my_classifier.save_weights('./model/model-bw.h5')
