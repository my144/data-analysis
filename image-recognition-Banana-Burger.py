#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50


# In[2]:


# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet50.ResNet50()


# In[3]:


# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img("banana.jpg", target_size=(224, 224))


# In[4]:


# Convert the image to a numpy array
x = image.img_to_array(img)


# In[5]:


# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)


# In[6]:


# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)


# In[7]:


# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)


# In[8]:


# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = resnet50.decode_predictions(predictions, top=7)


# In[9]:


print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))


# In[24]:


# Load another image file, resizing it to 224x224 pixels (required by this model)
img1 = image.load_img("burger1.jpg", target_size=(224, 224))


# In[25]:


# Convert the image to a numpy array
x1 = image.img_to_array(img1)


# In[26]:


# Add a forth dimension since Keras expects a list of images
x1 = np.expand_dims(x1, axis=0)


# In[27]:


# Scale the input image to the range used in the trained network
x1 = resnet50.preprocess_input(x1)


# In[28]:


# Run the image through the deep neural network to make a prediction
predictions = model.predict(x1)


# In[29]:


# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = resnet50.decode_predictions(predictions, top=9)


# In[30]:


print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))


# In[ ]:




