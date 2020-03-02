# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

#note: here no datapreprocessing as it is manually done by using directory structure
# in total - 10,000 images are there where cat = 5000 and dog = 5000
# training set - 8000 (cat =4k dog = 4k)
# test set - 2000 (cat = 1k   dog = 1k)

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import tensorflow
from tensorflow.keras.models import Sequential # to intiliaze our NN as sequence of layers n not graph
from tensorflow.keras.layers import Dense # ann begins, adding hidden layers (used for full connection)

from tensorflow.keras.layers import Conv2D  # convolution2D is for first step: convolution layers 2D for dealing with images
from tensorflow.keras.layers import MaxPooling2D # pooling step to add pooling layers
from tensorflow.keras.layers import Flatten # flattening step to convert to large feature vector for becoming input for ANN




# Initialising the CNN
classifier = Sequential() # creating object classifier acting as  model (cnn initiliazed)

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
'''
-convolution: take input image, apply many feature detectors and find feature map!
-no. of feature detectors will give same no. of feature maps
-conv2D is used and not dense but similar
-parameters: first arguement is 32 = nb_filter = no. of feature detectors (filters) used
32 and (3,3) means 32 feature detectors will be used and each filter will be 3rows- 3cols which will give rise to
32 feature maps (CPU is used so 32 is enough no need 64)
- default arg: border_node to handle the borders of input image (default value)
- input_shape: shape of input image on which we will apply feature detectors through convoltion
all our images do not have same size so we should force them into one single format and fixed size
(64,64,3) - 3 cuz 3d array for color images 64X64 pixels
- activation func - here relu to avoid negative pixel values



'''
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
'''
- pooling is reducing the size of the feature map! (so that we get less no. of nodess)
- pool_size is 1st arg: stride to move is 2X2 (commonly used)

'''

# Adding a second convolutional layer (for improving accuracy)
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
'''
- no parameters required, keras will understand using classifier
'''


# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
'''
- classic ann composed of fully connected layers
- dense function for adding fully connected layers. 128 is no. of hidden nodes (experimenting)
- another dense function for output: 1 node in the output layer telling dog/cat

'''












# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])





# Part 2 - Fitting the CNN to the images

from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''
- to generate image augmentation. avoids overfitting so we need many many images!
- we either need more images or we need ImagedataGenerator. it creates lot of batches and gives more images with
random transformations (like flipped, rotated, etc)
'''

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
'''
here we obtain different augmented pictures as we have less data
rescale: pixel values betweel  0 and 1, shear_range: random transformations zoom = random zooms, flipped images
'''

test_datagen = ImageDataGenerator(rescale = 1./255)
'''
here we only need to rescale.
'''

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
'''
- system directory to extract images from
- size of image expected in cnn model (before 64X64 only we gave in cnn)
- batches size who will undergo random transformations
- classes for output as we have dog and cat
'''

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
'''
- same as above
'''

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set)
'''
- we fit cnn to training set here
- no. ofimages in training set
- epochs = 1 complete dataset
- testset for valuating the performace
- no. of images in testset
'''
                         

print("success")


'''
classes = train_generator.class_indices    
>>> print(classes)
    {'cats': 0, 'dogs': 1}
    to know class indices



    to predict:
    import cv2
    img = cv2.imread('test.jpg')
img = cv2.resize(img,(320,240))
img = np.reshape(img,[1,320,240,3])

classes = model.predict_classes(img)

print classes
'''
