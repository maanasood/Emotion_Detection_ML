from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# PREPROCESSING
train_generator = train_data_gen.flow_from_directory('images_/train',
                                                     target_size=(48, 48),
                                                     batch_size=64,
                                                     color_mode='grayscale',
                                                     class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory('images_/validation',
                                                               target_size=(
                                                                   48, 48),
                                                               batch_size=64,
                                                               color_mode='grayscale',
                                                               class_mode='categorical')

# CREATE THE MODEL
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64, (3, 3), activation='relu',
          padding='same', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

# Fully connected layer 2nd layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=Adam(
    lr=0.0001), metrics=['accuracy'])

# TRAIN THE MODEL
checkpoint = ModelCheckpoint(
    "model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

model_info = model.fit_generator(train_generator,
                                 epochs=20,
                                 validation_data=validation_generator,
                                 callbacks=callbacks_list)

# SAVE THE MODEL IN JSON FILE
model_json = model.to_json()
with open("Emotion_model.json", 'w') as json_file:
    json_file.write(model_json)

# SAVE THE MODEL WEIGHTS
model.save_weights('Emotion_model.h5')
