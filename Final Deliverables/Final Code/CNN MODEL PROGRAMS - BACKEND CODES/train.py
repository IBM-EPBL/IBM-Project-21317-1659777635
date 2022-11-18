import tensorflow as tf
import trainlist
import os

TRAIN_DIR="./Data/Train"
VALIDATE_DIR="./Data/Test"
dataset=trainlist.dataset


def train_model():
    
    train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    train_generator=train_datagen.flow_from_directory(TRAIN_DIR,target_size=(224,224),class_mode="categorical",batch_size=300)

    validate_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    validate_generator=validate_datagen.flow_from_directory(VALIDATE_DIR,target_size=(224,224),class_mode="categorical",batch_size=300)

    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),activation="relu",input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128,(3,3),activation="relu",input_shape=(112,112)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(256,(3,3),activation="relu",input_shape=(56,156)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(512,(3,3),activation="relu",input_shape=(28,28)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(512,(3,3),activation="relu",input_shape=(14,14)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256,activation="relu"),
        tf.keras.layers.Dense(10,activation="softmax")

    ])

    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

    model.fit_generator(train_generator,epochs=10,validation_data=validate_generator,verbose=1,validation_steps=10,steps_per_epoch=20)

    model.save("sign_1.h5")
    
    print("Model Trained Sucessfully...")

def train_list(dataset):
    
    f=open("labels_1.txt","w+")
    
    for i in range(len(dataset)):
        f.write(str(i)+" "+dataset[i]+"\n")
    
    f.close()
    
    print("Dataset list as been sucessfully created")

train_model()
train_list(dataset)