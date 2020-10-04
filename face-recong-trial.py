#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PIL.Image
import PIL.ImageDraw
import face_recognition


# In[2]:


image = face_recognition.load_image_file("friend.jpg")


# In[3]:


# Find all the faces in the image
face_locations = face_recognition.face_locations(image)


# In[4]:


number_of_faces = len(face_locations)
print("I found {} face(s) in this photograph.".format(number_of_faces))


# In[5]:


# Load the image into a Python Image Library object so that we can draw on top of it and display it
pil_image = PIL.Image.fromarray(image)


# In[6]:


for face_location in face_locations:

    # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # Let's draw a box around the face
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([left, top, right, bottom], outline="red")


# In[7]:


# Display the image on screen
pil_image.show()


# In[ ]:




