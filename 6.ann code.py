# Artificial Neural Network



# ------------------------------------------------------Part 1 - Data Preprocessing------------------------------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv') #total 13 + 1 cols
X = dataset.iloc[:, 3:13].values  # take X features(columns) from 3rd col to 12th col 
y = dataset.iloc[:, 13].values # last 14th col output

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()    # encoding countries {France, spain, germany}
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # all rows of 1st col {0,1,2}
labelencoder_X_2 = LabelEncoder()  # encoding gender {m ,f} 
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # all rows of 2nd col  {0,1}

# using OneHotEncoder for country as 3 or more values {0,1,2}
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)


X = transformer.fit_transform(X.tolist()) # 3 seperate cols for Frnce, Spain, Germany created
X = X[:, 1:] # exclude france col here


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



















# ------------------------------------------------------Part 2 - Now let's make the ANN!------------------------------------------------------

# Importing the Keras libraries and packages
import tensorflow
from tensorflow.keras.models import Sequential # keras will build NN based on Tf background
from tensorflow.keras.layers import Dense # seq model used for initializing and dense for layers in NN

# Initialising the ANN are 2 ways: graphic/ sequence of layers (here sequence of layers)
classifier = Sequential() # here our problem is classification so ann model is a classfier


#step1) weights initialisation  taken care by dense
# step2) first row is going to input layer (11 independent variable so 11 nodes)
# step3) higher the value of AF,more impact it is going to have in the network, more it will pass signal from L->R
# here we are choosing Rectifier (relu) based on research its best for hidden layer. Also sigmoid is good for output layer cuz we get probabilities (ranking too)
#step4) the actual and pred are compared and error is found
#step5) This error is backpropogated from R->L and weights are re initialised acc to how much they are responsible for the generation of error
#step6) repeating above steps
#steps7) epoch








# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='glorot_uniform' , activation = 'relu', input_dim = 11)) # outputdim = noo .of nodes in hidden layer is 11+1/2=6
# uniform func initialises weights close to 0
# AF = rectifier for hidden & input nodes = 11 variables in 1st row here

# Adding the second hidden layer (no need of input_dim as its same here)
classifier.add(Dense(units= 6, kernel_initializer='glorot_uniform' , activation = 'relu'))

# so here we are using 2 hidden layers

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer='glorot_uniform', activation = 'sigmoid')) # 1 node for output, Af = sigmoid for 2 output categories
# softMax for more than 2 o/p categories

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# optimizer = algo for finding best weights for powerful NN (here adam: stochastic gradient descent algo type)
# loss = loss within adam algo (like sum of sqred differences for cost function but here log type(binary_classentropy) for 2 categories)
# if 3 or more o/p then loss = "categorical_crossentropy"
# mertrics = here accuracy to evaluate our model


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# batch_size 10 means no. of observations after which we want to update weights
# nb_epoch = whole training set passed then epoch becomes 1
# upon executing, 86% accuracy converged.. we reached convergence
 







# ------------------------------------------------------Part 3 - Making the predictions and evaluating the model------------------------------------------------------

# Predicting the Test set results
y_pred = classifier.predict(X_test) # a col of all probabilities obtained!!
y_pred = (y_pred > 0.5)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
