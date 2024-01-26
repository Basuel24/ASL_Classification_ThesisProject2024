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
                    #modelCNN = tf.keras.models.load_model("CNN/50epoch_CNN_model.h5")
                    #modelCNN = tf.keras.models.load_model("CNN/cnn_model.h5")
                    modelCNN = tf.keras.models.load_model("CNN/80 epochs/cnn_model.h5")
                    
                    
                    predictionCNN = modelCNN.predict(image_input)
                    MaxPositionCNN=np.argmax(predictionCNN)  
                    confidenceCNN = round(predictionCNN[0,MaxPositionCNN]*100, 1)
                    predictionCNN_label=class_names[MaxPositionCNN]
                    
                    #SA
                    modelSA = tf.keras.models.load_model("SA/80 epochs/sa_model.h5")
                    #modelSA = tf.keras.models.load_model("SA/sa_model.h5")
                    
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
##################################################################
    alphab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapping_letter = {}
    count = 0
##################################################################
    train_df = pd.read_csv('dataset/train.csv')
    train_df.rename(columns={'label':'Label'},inplace = True)
    train_df = train_df.sample(frac = 1.0).reset_index(drop = True)
##################################################################
    for i,l in enumerate(alphab):
        mapping_letter[l] = i
    mapping_letter = {v:k for k,v in mapping_letter.items()}
##################################################################
    def to_image(array, label=True):
        # Convert the input array to a NumPy array
        array = np.array(array)
    
        # Check if the array is empty
        if array.size == 0:
            print("Error: Empty array.")
            return None  # or handle the error in an appropriate way
    
        # Set the start index based on whether a label is included
        start_idx = 1 if label else 0
    
        # Calculate the expected size after considering the start index
        expected_size = 28 * 28
    
        # Check if the array has enough elements to reshape
        if array[start_idx:].size < expected_size:
            print(f"Error: Insufficient elements in the array to reshape into {expected_size}-dimensional image.")
            return None  # or handle the error in an appropriate way
    
        # Reshape the array into a (28, 28) image format
        result = array[start_idx:].reshape(28, 28).astype(float)
        return result
    
    #result_image = to_image(your_array, label=True)
##################################################################
    # Display some pictures of the dataset
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(12, 12),
                            subplot_kw={'xticks': [], 'yticks': []})
##################################################################
    #modelCNN = tf.keras.models.load_model("CNN/100CNN_model.h5")
    #modelCNN = tf.keras.models.load_model("CNN/80 epochs/cnn_model.h5")
    modelCNN = tf.keras.models.load_model("SA/80 epochs/sa_model.h5")
##################################################################
    #               1   2   3   4   5   6   7   8   9  {J} 10  11  12  13  14  15  16  17  18  19  20  21  22  23  24 {Z}
    class_names = ['A','B','C','D','E','F','G','H','I','','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','']
    a = 0 
    for i, ax in enumerate(axes.flat):        
        inputImage = to_image(train_df.iloc[i])
        img = inputImage
        image_input = asarray(inputImage)
        image_input = np.expand_dims(image_input, axis = 0)
        prediction = modelCNN.predict(image_input)
        MaxPositionindex=np.argmax(prediction, axis=1)
        title = mapping_letter[train_df.Label[i]]
        
        indices = np.where(prediction>= 0.0001)
        arr = prediction[prediction>= 0.0001]  
        
        if (title == class_names[MaxPositionindex[0]]):
            ax.set_title("T: ("+title+") P: ("+class_names[MaxPositionindex[0]]+") ", fontsize = 15).set_color('black')
            ax.imshow(img)
            for j in range(len(arr)):
                ax.set_ylabel("%0.2f%%" % (arr[j] * 100))
                ax.set_xlabel("correct")
            count += 1
        else:
            ax.set_title("T: ("+title+") P: ("+class_names[MaxPositionindex[0]]+") ", fontsize = 15).set_color('red')
            ax.imshow(img, cmap = 'gray')
            for j in range(len(arr)):
                ax.set_ylabel("%0.2f%%" % (arr[j] * 100))
                ax.set_xlabel("wrong").set_color('red')
##################################################################
    st.markdown("<h5 style='text-align: left;'> Predicted = ("+ str(count) +"/40) </h5>", unsafe_allow_html=True)
##################################################################
    plt.tight_layout(pad=0.5)
    #plt.show()
    st.pyplot(fig)
################################################################################################
    #end of CNN Baseline Prediction
################################################################################################
with tabs[2]:
    count = 0
    # Display some pictures of the dataset
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(12, 12),
                            subplot_kw={'xticks': [], 'yticks': []})
##################################################################
    #modelSA = tf.keras.models.load_model("SA/50epoch_CNN_Model.h5")
    #modelSA = tf.keras.models.load_model("SA/1new50SA_model.h5")
    modelSA = tf.keras.models.load_model("CNN/80 epochs/cnn_model.h5")
    #modelSA = tf.keras.models.load_model("SA/HorizontalFlipFalse50SA_model.h5")
    #               1   2   3   4   5   6   7   8   9      10  11  12  13  14  15  16  17  18  19  20  21  22  23  24 
    class_names = ['A','B','C','D','E','F','G','H','I','','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    for i, ax in enumerate(axes.flat):
        inputImage = to_image(train_df.iloc[i])
        img = inputImage
        image_inputSA = asarray(inputImage)
        image_inputSA = np.expand_dims(image_inputSA, axis = 0)
        predictionSA = modelSA.predict(image_inputSA)
        
        MaxPositionindex=np.argmax(predictionSA, axis=1)        
        title = mapping_letter[train_df.Label[i]]
        
        indices = np.where(prediction>= 0.0001)
        arr = prediction[prediction>= 0.0001]  
        
        if (title == class_names[MaxPositionindex[0]]):
            ax.set_title("T: ("+title+") P: ("+class_names[MaxPositionindex[0]]+")", fontsize = 15).set_color('black')
            ax.imshow(img, cmap = 'Oranges_r')
            for j in range(len(arr)):
                ax.set_ylabel("%0.2f%%" % (arr[j] * 100))
                ax.set_xlabel("correct")
            count += 1
        else:
            ax.set_title("T: ("+title+") P: ("+class_names[MaxPositionindex[0]]+")", fontsize = 15).set_color('red')
            ax.imshow(img, cmap = 'gray')
            for j in range(len(arr)):
                ax.set_ylabel("%0.2f%%" % (arr[j] * 100))
                ax.set_xlabel("wrong").set_color('red')
##################################################################
    st.markdown("<h5 style='text-align: left;'> Predicted = ("+ str(count) +"/40) </h5>", unsafe_allow_html=True)
##################################################################
    plt.tight_layout(pad=0.5)
    #plt.show()
    st.pyplot(fig)
################################################################################################
    #end of CNN with Style Augmentation Prediction
################################################################################################
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
