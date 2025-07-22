# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:59:37 2025

@author: polas
"""
import streamlit as st
from PIL import Image, ImageDraw , ImageFont
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')
object_names = list(model.names.values())


img_file_buffer = st.camera_input("Take a picture", width ='stretch')

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Check the type of img_array:
    # Should output: <class 'numpy.ndarray'>
    

    result = model.predict(img, verbose=True)
    for detection in result[0].boxes.data:
       x0, y0 = (int(detection[0]), int(detection[1]))
       x1, y1 = (int(detection[2]), int(detection[3]))
       score = round(float(detection[4]), 2)
       cls = int(detection[5])
       object_name =  model.names[cls]
       label = f'{object_name} {score}'  
      
       if  object_name != '' :
           img1 = ImageDraw.Draw(img)  
           img1.rectangle([x0, y0, x1, y1] , outline ="red" , width=3)
           draw = ImageDraw.Draw(img)  
           font = ImageFont.truetype("ARIAL.TTF", 20 )
           draw.text( (x0, y0-20), label, font=font , fill=(255,0,0))
       else :
           pass
                    
    
    st.image(img, channels="RGB", use_container_width= "auto")
