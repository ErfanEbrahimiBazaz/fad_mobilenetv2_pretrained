from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2 as mbv2
# from keras.applications.nasnet import NASNetMobile
# from keras_applications.resnet50 import ResNet50
import keras.applications
import time
import cv2 as cv
# import Config


pretrain = "./myClassifier/mobilenetV2/train_1/" + "saved-model_mbNetV2-59.hdf5"
trainFromScratch = False

if trainFromScratch is True:
    model = keras.applications.mobilenet_v2.MobileNetV2
    # base_model = ResNet50(weights='imagenet', include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
    base_model = mbv2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
else:
    # load pretrained Model
    model = keras.models.load_model(pretrain)
    base_model = model

model.summary()
# IMG_SIZE = 75
imageSize        = 224
imageChannels     = 3
batchSize      = 32
numberOfTrainImages    = 108974
numberOfValidationImages    = 20759

train_datagen = ImageDataGenerator(
    rotation_range      = 1,
    # width_shift_range   =0.2,
    # height_shift_range  =0.2,
    rescale = 1./255.,
    # shear_range = 0.2,
    # zoom_range = 0.2,
    # horizontal_flip = True
)
validation_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(
    'D:/ClassifyByType_1Car/Train',
    target_size=(imageSize, imageSize),
    class_mode='categorical',
    batch_size=batchSize,)

validation_generator = validation_datagen.flow_from_directory(
    'D:/ClassifyByType_1Car/valid',   #train
    target_size=(imageSize, imageSize),
    class_mode='categorical',
    batch_size=batchSize)

if trainFromScratch is True:
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(32, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    # x = Dense(64, activation='relu')(x)  # dense layer 2
    # x = Dense(128, activation='relu')(x)  # dense layer 3
    preds = Dense(35, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    # for layer in model.layers:
    #     layer.trainable = True

    # or if we want to set the first 20 layers of the network to be non-trainable
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

# ------------------------------------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

step_size_train = train_generator.n // train_generator.batch_size
filepath = "./myClassifier/mobileNetV2/" + "saved-model_mbNetV2-{epoch:02d}.hdf5"

checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             verbose=1,
                                             mode='auto',
                                             monitor='accuracy',
                                             save_best_only=True
                                             )

callbacks_list = [checkpoint]

start = time.time()

# def fit_generator(self, generator,
#                       steps_per_epoch=None,
#                       epochs=1,
#                       verbose=1,
#                       callbacks=None,
#                       validation_data=None,
#                       validation_steps=None,
#                       class_weight=None,
#                       max_queue_size=10,
#                       workers=1,
#                       use_multiprocessing=False,
#                       shuffle=True,
#                       initial_epoch=0):

model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    validation_data=validation_generator,
                    validation_steps=1,
                    epochs=1000,
                    callbacks=callbacks_list,
                    shuffle=True)

end = time.time()
print('Processing time:', (end - start) / 60)
model.save_weights('cnn.h5')
