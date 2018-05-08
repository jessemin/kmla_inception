from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np
import os

# load a pretrained InceptionResnetV2
model = InceptionResNetV2(weights='imagenet')

while True:
    # get input from user
    file_name = raw_input("Enter the image file name: ")
    if file_name == "":
        break
    # get the path to the target image
    img_path = os.path.join('imgs', file_name)
    try:
        # read the image from the file
        img = image.load_img(img_path, target_size=(299, 299))
    except:
        print "ERROR: File doesn't exist!"
        continue
    # preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # ask model to classify
    preds = model.predict(x)

    # print the top 3 predictions
    print('Input image was: {}'.format(file_name))
    print('Predicted:', decode_predictions(preds, top=3)[0])
