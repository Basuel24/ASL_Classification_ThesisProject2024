import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.io import imread, imshow
from skimage.transform import resize
from numpy import asarray
from tensorflow.keras.models import load_model
st.markdown("<h4 style='text-align: center;'> Enhance Convolutional Newral Network (CNN) for Image Classification of Alphabet Sign Language (ASL) </h4>", unsafe_allow_html=True)
    
################################################################################################
tab_title = [
    "Test Image",
    "Baseline (CNN)",
    "Applied Algorithm (CNN with Augmentation)",
    "Algorithms Graph"
]
tabs = st.tabs(tab_title)

if "load_state" not in st.session_state:
    st.session_state.load_state = False

with tabs[0]:
        #               1   2   3   4   5   6   7   8   9      10  11  12  13  14  15  16  17  18  19  20  21  22  23  24 
    class_names = ['A','B','C','D','E','F','G','H','I','','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    ##########################################################################################
    uploaded_files = st.file_uploader("", type=["jpg"], accept_multiple_files=True)
    if uploaded_files or st.session_state.load_state:
        st.session_state.load_state = True
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name == '':
                    st.write('')
                else:
                    image = Image.open(uploaded_file)
                    new_image = image.resize((50, 50))
                    st.image(new_image, caption="Image Preview")
                    
                    image_input = asarray(image)
                    image_input = resize(image_input, (28, 28, 1))
                    image_input = np.expand_dims(image_input, axis = 0)
                    #CNN
                    modelCNN = tf.keras.models.load_model("CNN/80 epochs/cnn_model.h5")
                    predictionCNN = modelCNN.predict(image_input)
                    MaxPositionCNN=np.argmax(predictionCNN)  
                    confidenceCNN = round(predictionCNN[0,MaxPositionCNN]*100, 1)
                    predictionCNN_label=class_names[MaxPositionCNN]
                    
                    #SA
                    modelSA = tf.keras.models.load_model("SA/80 epochs/sa_model.h5")
                    predictionSA = modelSA.predict(image_input) 
                    MaxPositionSA=np.argmax(predictionSA)  
                    confidenceSA = round(predictionSA[0,MaxPositionSA]*100, 1)
                    predictionSA_label=class_names[MaxPositionSA]
                    
                    indicesCNN = np.where(predictionCNN>= 0.0001)
                    arrCNN = predictionCNN[predictionCNN>= 0.0001]
                    
                    indicesSA = np.where(predictionSA>= 0.0001)
                    arrSA = predictionSA[predictionSA>= 0.0001]
                    
                    ############################################################
                    #selection sort descending order using CNN
                    temp = 0
                    tempindex = 0
                    for i in range(0, len(arrCNN)):
                        for j in range(i+1, len(arrCNN)):
                            if arrCNN[i] < arrCNN[j]:
                                temp = arrCNN[i]
                                tempindex = indicesCNN[1][i]
                                
                                arrCNN[i] = arrCNN[j]
                                indicesCNN[1][i] = indicesCNN[1][j]
                                
                                arrCNN[j] = temp
                                indicesCNN[1][j] = tempindex
                    ############################################################

                    ############################################################
                    #selection sort descending order using SA
                    tempSA = 0
                    tempindexSA = 0
                    for i in range(0, len(arrSA)):
                        for j in range(i+1, len(arrSA)):
                            if arrSA[i] < arrSA[j]:
                                tempSA = arrSA[i]
                                tempindexSA = indicesSA[1][i]
                                
                                arrSA[i] = arrSA[j]
                                indicesSA[1][i] = indicesSA[1][j]
                                
                                arrSA[j] = tempSA
                                indicesSA[1][j] = tempindexSA
                    ############################################################
                    ################################################################################################
                    #create a graph using the predicted  accuracy data
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize = (12, 8))
                    plt.subplot(2, 2, 1)
                    plt.ylabel('Accuracy')
                    plt.xlabel('Letters')
                    data = []
                    if len(arrCNN)<5:
                        for i in range(len(arrCNN)):
                            langs = [class_names[indicesCNN[1][i]]]
                            students = arrCNN[i]
                            percentage = " %0.2f%%" % (arrCNN[i] * 100)
                            plt.bar(langs,students, bottom=None, align='center', data=None)
                            plt.text(langs,students,percentage, ha = 'center')
                    else:
                            for i in range(5):
                                langs = [class_names[indicesCNN[1][i]]]
                                students = arrCNN[i]
                                percentage = " %0.2f%%" % (arrCNN[i] * 100)
                                plt.bar(langs,students, bottom=None, align='center', data=None)
                                plt.text(langs,students,percentage, ha = 'center')
                    plt.legend()
                    plt.title('CNN Prediction Graph')
                                        
                    plt.subplot(2, 2, 2)
                    if len(arrSA)<5:
                        for i in range(len(arrSA)):
                            langs = [class_names[indicesSA[1][i]]]
                            students = arrSA[i]
                            percentage = " %0.2f%%" % (arrSA[i] * 100)
                            plt.bar(langs,students, bottom=None, align='center', data=None)
                            plt.text(langs,students,percentage, ha = 'center')
                    else:
                            for i in range(5):
                                langs = [class_names[indicesSA[1][i]]]
                                students = arrSA[i]
                                percentage = " %0.2f%%" % (arrSA[i] * 100)
                                plt.bar(langs,students, bottom=None, align='center', data=None)
                                plt.text(langs,students,percentage, ha = 'center')
                    plt.legend()
                    plt.title('CNN_SA Prediction Graph')
                    st.pyplot(fig)
                    ################################################################################################
                    #Letters = ['A','B','C','D','E','F','G','H','I','','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']                    
                ##########################################################################################

    
################################################################################################
with tabs[1]:
    import pandas as pd
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    
    # Load the dataset
    train = pd.read_csv('dataset/sign_mnist_train.csv')
    
    # Identify and handle non-numeric values in the DataFrame
    non_numeric_mask = ~train.applymap(np.isreal).all(1)
    if non_numeric_mask.any():
        print("Handling non-numeric values in the dataset.")
        print("Rows with non-numeric values:")
        print(train[non_numeric_mask])
        train = train[~non_numeric_mask]
    
    # Exclude the 'label' column from the conversion to NumPy array
    train_set = np.array(train.drop(columns=['label']), dtype='float32')
    
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Define the dimensions of the plot grid
    W_grid = 5
    L_grid = 5
    fig, axes = plt.subplots(L_grid, W_grid, figsize=(10, 10))
    axes = axes.ravel()  # flatten the 5x5 matrix into a 1D array
    n_train = len(train_set)
    
    # Select a random number from 0 to n_train
    for i in np.arange(0, W_grid * L_grid):
        # Select a random number
        index = np.random.randint(0, n_train)
        
        # Read and display an image with the selected index
        axes[i].imshow(train_set[index].reshape((28, 28)))
        
        label_index = int(train['label'][index])  # Access the label from the original DataFrame
        axes[i].set_title(class_names[label_index], fontsize=8)
        axes[i].axis('off')
    
    plt.subplots_adjust(hspace=0.4)
    plt.show()


################################################################################################
    #end of CNN Baseline Prediction
################################################################################################
with tabs[2]:
    
    import random

################################################################################################
    #end of CNN Baseline Prediction
with tabs[3]:
    ################################################################################################
    #CNN Graph
    st.markdown("<h5 style='text-align: Center;'> Baseline (CNN) </h5>", unsafe_allow_html=True)
    ################################################################################################
    #load the CNN training validation accuracy to local folder 
    #val_accuracy = np.load('CNN/100CNN_val_accuracy.npy', allow_pickle='TRUE').item()
    val_accuracy = np.load('CNN/80 epochs/SAValidationAccuracy.npy', allow_pickle='TRUE').item()
    ##################################################################
    #load the CNN training history from local folder
    #history = np.load('CNN/100CNN_history.npy', allow_pickle='TRUE').item()
    history = np.load('CNN/80 epochs/history.npy', allow_pickle='TRUE').item()
    ##################################################################
    #Plotting the accuracy training using graph presentation CNN
    fig = plt.figure(figsize = (12, 8))
    #plt.plot([4, 2, 1, 3, 5])
    plt.subplot(2, 2, 1)
    plt.ylabel('Loss Value')
    plt.xlabel('Epochs')
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.title('Loss Graph')
    #Plotting the accuracy training using graph presentation CNN
    plt.subplot(2, 2, 2)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(history['accuracy'], label='Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.title('Accuracy Graph')
    st.pyplot(fig)
    st.markdown("<h5 style='text-align: Left;'> Predictions </h5>", unsafe_allow_html=True)

    training_accuracy = float(history['accuracy'][-1])
    st.markdown("<h5 style='text-align: Left;'>" '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Training Accuracy: %0.2f%%' % (training_accuracy * 100) + "</h5>", unsafe_allow_html=True)
    #st.title('Predictions: %0.2f%%' % (training_accuracy * 100))

    validation_accuracy = float(history['val_accuracy'][-1])
    st.markdown("<h5 style='text-align: Left;'>" '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Validation Accuracy: %0.2f%%' % (validation_accuracy * 100) + "</h5>", unsafe_allow_html=True)
################################################################################################
    #end of CNN Graph
################################################################################################

################################################################################################
    #Style Augmentation Graph
    st.markdown("<h5 style='text-align: center;'> Applied Algorithm (CNN with Augmentation)</h5>", unsafe_allow_html=True)
################################################################################################
    #load the CNN training validation accuracy to local folder 
    #val_accuracy = np.load('SA/100SA_val_accuracy.npy', allow_pickle='TRUE').item()
    val_accuracy = np.load('SA/80 epochs/SAValidationAccuracy.npy', allow_pickle='TRUE').item()
    ##################################################################
    #load the CNN training history from local folder
    #history = np.load('SA/100SA_history.npy', allow_pickle='TRUE').item()
    history = np.load('SA/80 epochs/history.npy', allow_pickle='TRUE').item()
    ##################################################################
    #Plotting the accuracy training using graph presentation CNN
    fig = plt.figure(figsize = (12, 8))
    #plt.plot([4, 2, 1, 3, 5])
    plt.subplot(2, 2, 1)
    plt.ylabel('Loss Value')
    plt.xlabel('Epochs')
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.title('Loss Graph')
    #Plotting the accuracy training using graph presentation CNN
    plt.subplot(2, 2, 2)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(history['accuracy'], label='Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.title('Accuracy Graph')
    st.pyplot(fig)
    
    training_accuracy = float(history['accuracy'][-1])
    st.markdown("<h5 style='text-align: Left;'>" '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Training Accuracy: %0.2f%%' % (training_accuracy * 100) + "</h5>", unsafe_allow_html=True)
    #st.title('Predictions: %0.2f%%' % (training_accuracy * 100))

    validation_accuracy = float(history['val_accuracy'][-1])
    st.markdown("<h5 style='text-align: Left;'>" '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Validation Accuracy: %0.2f%%' % (validation_accuracy * 100) + "</h5>", unsafe_allow_html=True)
################################################################################################
    #end of CNN with Style Augmentation Graph
################################################################################################
