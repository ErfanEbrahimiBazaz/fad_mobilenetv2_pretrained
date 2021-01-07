from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2 as mbv2
import keras.applications
import time
import os
import matplotlib.pyplot as plt


# Defining variables
imageSize = 224
imageChannels = 3
batchSize = 32
numberOfTrainImages = 108974
numberOfValidationImages = 20759
# paths
base_dir = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\kaggle\prc'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# loading model
model = keras.applications.mobilenet_v2.MobileNetV2
base_model = mbv2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, )

validation_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(imageSize, imageSize),
    class_mode='categorical',
    batch_size=batchSize
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(imageSize, imageSize),
    class_mode='categorical',
    batch_size=batchSize
)

x = base_model.output
# print(x) # Tensor("out_relu/Relu6:0", shape=(None, 7, 7, 1280), dtype=float32)
# print(base_model.input) # Tensor("input_1:0", shape=(None, 224, 224, 3), dtype=float32)
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
# x = Dense(64, activation='relu')(x)  # dense layer 2
# x = Dense(128, activation='relu')(x)  # dense layer 3
preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

for i, layer in enumerate(model.layers):
    print(i, layer.name)

for layer in model.layers[:155]:
    layer.trainable = False

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

print('Training size = {},  batch size = {}'.format(train_generator.n, train_generator.batch_size) )
step_size_train = train_generator.n // train_generator.batch_size
print('Step size is {}'.format(step_size_train))

filepath = "./myClassifier/mobileNetV2/" + "saved-model_mbNetV2-{epoch:02d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             verbose=1,
                                             mode='auto',
                                             monitor='accuracy',
                                             save_best_only=True
                                             )
callbacks_list = [checkpoint]
start = time.time()

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_size_train, # steps_per_epoch = len(X_train)//batch_size,
                              validation_data=validation_generator,
                              validation_steps=31, # validation_steps = total_validation_samples//validation_batch_size
                              epochs=60,
                              callbacks=callbacks_list,
                              shuffle=True)

end = time.time()
print('Processing time:', (end - start) / 60)
model.save_weights('cnn.h5')

# binary classifier with 12000 images with 2 added new hidden layers with drop-out layers at layers 1 and 2.
model.save('bincls_mobileNetV2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy drop-out at layers 1 and 2')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss drop-out at layers 1 and 2')
plt.legend()

plt.show()
