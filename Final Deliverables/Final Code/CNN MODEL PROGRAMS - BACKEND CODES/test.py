import tensorflow as tf
import numpy as np
import trainlist

model=tf.keras.models.load_model("./Model/keras_model.h5")
image=tf.keras.preprocessing.image
#print(model.summary())

fl_img='./Data/Train/D/Image_1667714863.1011443.jpg' 
img=image.load_img(fl_img,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=np.argmax(model.predict(x))
op=trainlist.dataset
ans=op[pred]
print("\n\t"+ans+"\n")