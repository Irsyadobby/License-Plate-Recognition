import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json
from ultralytics import YOLO
import pandas as pd
import io


# Configurations
CFG_MODEL_PATH = "models/best submit 2.pt"
CFG_ENABLE_URL_DOWNLOAD = True
CFG_ENABLE_VIDEO_PREDICTION = True
if CFG_ENABLE_URL_DOWNLOAD:
    # Configure this if you set cfg_enable_url_download to True
    url = "https://archive.org/download/yoloTrained/yoloTrained.pt"
# End of Configurations

model = YOLO('model/best.pt')

# Initialize Session State
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'image_outputs' not in st.session_state:
    st.session_state.image_outputs = []
if 'output_tab' not in st.session_state:
    st.session_state.output_tab = False
if 'input_type' not in st.session_state:
    st.session_state.input_type = "Single"



def imageInput(model, src, input_type):
    if src == 'External Data':

        if st.session_state.input_type != input_type :
            st.session_state.results_df = None
            st.session_state.image_outputs = []
            st.session_state.output_tab = False
            st.session_state.input_type = input_type
                
        if input_type == 'Multiple' :
            images = st.file_uploader('Select Images', type = ['jpg','png','jpeg'], accept_multiple_files = True)

            if images:
                st.write('Uploaded Files :')

                filenames = []
                for file in images :
                    filenames.append(file.name)
                
                df_image = pd.DataFrame({'Name of File' : filenames})
                st.dataframe(df_image)

            if st.button('Predict!'):
                total_images = len(images)
                result_list = []

                for idx, uploaded_image in enumerate(images):
                    image = Image.open(uploaded_image)

                    result = model.predict(image, hide_conf=True, line_width=1, retina_masks=True, iou=0.5, augment=True, max_det=9, agnostic_nms=True)
                    result = result[0]
                    prediction_json = result.tojson()
                    predictions = json.loads(prediction_json)

                    with open('data/predefined_classes.txt', 'r') as f:
                        classes = f.read().split()
                    
                    filtered_predictions = [pred for pred in predictions if pred['name'] in classes]
                    sorted_predictions = sorted(filtered_predictions, key=lambda x: x['box']['x1'])
                    combined_classes = ''.join([pred['name'] for pred in sorted_predictions])

                    filename = uploaded_image.name
                    result_list.append({'Name of File' : filename, 'Prediction' : combined_classes})

                    st.session_state.image_outputs.append((uploaded_image, sorted_predictions))
                
                st.subheader('Model Prediction :')

                st.session_state.results_df = pd.DataFrame(result_list)
                st.dataframe(st.session_state.results_df)
            
            if st.session_state.output_tab is not None and st.session_state.results_df is not None :
                csv_buffer = io.StringIO()
                st.session_state.results_df.to_csv(csv_buffer, index = False)
                csv_bytes = csv_buffer.getvalue()
                st.download_button('Download Prediction (csv)', data = csv_bytes, file_name = "Model Prediction.csv", mime = 'data/csv')
            

            if st.session_state.output_tab is not None :
                if st.button('Reset Prediction') :
                    st.session_state.results_df = None
                    st.session_state.image_outputs = []
                    st.session_state.output_tab = False
                
                st.warning("Make sure you click 'Reset Prediction' before doing another prediction !")
     
        
        if input_type == 'Single' :
            image_file = st.file_uploader(
                "Upload An Image", type=['png', 'jpeg', 'jpg'])
            col1, col2 = st.columns(2)
            if image_file is not None:
                img = Image.open(image_file)
                with col1:
                    st.image(img, use_column_width='always')

                with st.spinner(text = 'Predicting...'):
                    results = model.predict(img, hide_conf=True, line_width=1, retina_masks=True, iou=0.5, augment=True, max_det=9, agnostic_nms=True)
                    result = results[0]
                    prediction_json = result.tojson()
                    predictions = json.loads(prediction_json)

                    with open('data/predefined_classes.txt', 'r') as f:
                        classes = f.read().split()

                    filtered_predictions = [pred for pred in predictions if pred['name'] in classes]
                    sorted_predictions = sorted(filtered_predictions, key=lambda x: x['box']['x1'])
                    combined_classes = ''.join([pred['name'] for pred in sorted_predictions])

                    st.subheader("Model Prediction : ")
                    st.metric(label = "License Plate", value = combined_classes)

                with col2:
                    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100))
                    ax.imshow(img)
                    for pred in sorted_predictions:
                        box = pred['box']
                        rect = plt.Rectangle((box['x1'], box['y1']), box['x2'] - box['x1'], box['y2'] - box['y1'],
                                            fill=False, color='blue', linewidth=2)
                        ax.add_patch(rect)
                        font_size = int(img.width / 80)  # Adjust font size proportionally
                        plt.text(box['x1'], box['y1'], pred['name'], color='black', backgroundcolor='white', fontsize=font_size)
                    plt.axis('off')
                    st.pyplot(fig, bbox_inches='tight', pad_inches=0)


    elif src == 'Test Dataset':
        # Image selector slider
        imgpaths = glob.glob('data/example_images/*')
        if len(imgpaths) == 0:
            st.write(".")
            st.error(
                'No images found, Please upload example images in data/example_images', icon="")
            return
        imgsel = st.slider('Select images from Testing Dataset',
                           min_value=1, max_value=len(imgpaths), step=1)
        image_file = imgpaths[imgsel-1]
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                with st.spinner(text = "Predicting..."):
                    results = model.predict(img, hide_conf=True, line_width=1, retina_masks=True, iou=0.5, augment=True, max_det=9, agnostic_nms=True)
                    result = results[0]
                    prediction_json = result.tojson()
                    predictions = json.loads(prediction_json)

                    with open('data/predefined_classes.txt', 'r') as f:
                        classes = f.read().split()

                    filtered_predictions = [pred for pred in predictions if pred['name'] in classes]
                    sorted_predictions = sorted(filtered_predictions, key=lambda x: x['box']['x1'])
                    combined_classes = ''.join([pred['name'] for pred in sorted_predictions])

                    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100))
                    ax.imshow(img)
                    for pred in sorted_predictions:
                        box = pred['box']
                        rect = plt.Rectangle((box['x1'], box['y1']), box['x2'] - box['x1'], box['y2'] - box['y1'],
                                        fill=False, color='blue', linewidth=2)
                        ax.add_patch(rect)
                        font_size = int(img.width / 80)  # Adjust font size proportionally
                        plt.text(box['x1'], box['y1'], pred['name'], color='black', backgroundcolor='white', fontsize=font_size)
                    plt.axis('off')
                    st.pyplot(fig, bbox_inches='tight', pad_inches=0)
                
                st.subheader("Model Prediction : ")
                st.metric(label = 'License Plate', value = combined_classes)


def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        print('')
        
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error(
                'Model not found, please config if you wish to download model from url set `cfg_enable_url_download = True`  ', icon="⚠️")

    st.set_page_config(initial_sidebar_state = 'collapsed')

    st.title("Big Data Challenge Model Deployment")

    tab1, tab2 = st.tabs(["Model Description", "Prediction"])

    with tab1:
        with st.expander("What We Do") :
            st.subheader("Object Detection")
            st.write("""
            Object detection merupakan teknik dalam pengolahan citra dan penglihatan komputer untuk mengidentifikasi dan melokalisasi objek-objek tertentu dalam gambar atau video. Tujuan utama dari object detection adalah mengenali dan menempatkan kotak pembatas (bounding box) di sekitar objek-objek yang ada dalam gambar. Teknik ini menggabungkan konsep dari deteksi objek (mengenali objek mana yang ada) dan segmentasi semantik (mengenali di mana objek tersebut berada).

            Salah satu pemanfaatan object detection yang umum digunakan adalah pengidentifikasian plat nomor dalam gambar dengan menempatkan kotak pembatas (bounding box) di sekitar area plat nomor. Algoritma object detection untuk plat nomor umumnya menggunakan teknik deep learning, terutama dengan menggunakan model YOLOv5xu yang telah dilatih sebelumnya untuk mengidentifikasi dan mendeteksi plat nomor pada gambar atau video.""")


        with st.expander("Model In Use") :
            st.subheader("YOLOv5xu")
            gambar_yolo = Image.open('assets/yolo.png')
            yolo = gambar_yolo.resize((400,400))
            st.image(yolo)
            st.write("""
            YOLOv5xu adalah varian yang dikembangkan dari model YOLOv5x (You Only Look Once version 5x) yang dioptimalkan khusus untuk tugas deteksi plat nomor pada kendaraan. Konsep dasar YOLOv5xu tetap mengadopsi prinsip "You Only Look Once", di mana model dapat mendeteksi objek dengan cepat dan efisien dalam satu tahap. Dalam konteks deteksi plat nomor, YOLOv5xu mampu memberikan kemampuan yang ditingkatkan dalam mengenali plat nomor pada berbagai situasi dan kondisi visual.

            YOLOv5xu dirancang untuk memberikan efisiensi tinggi dalam melakukan deteksi plat nomor. Dengan pendekatan deteksi dalam satu tahap, model ini dapat mengidentifikasi lokasi dan kelas plat nomor dengan cepat tanpa memerlukan tahap tambahan. Model ini juga mampu melakukan deteksi pada berbagai skala objek. Hal ini sangat berguna dalam deteksi plat nomor pada kendaraan dengan ukuran yang beragam, baik dari jarak dekat maupun jauh. Dengan didukung oleh penggunaan teknologi terkini, YOLOv5xu memiliki tingkat akurasi yang tinggi dalam mengenali plat nomor. Hal ini memungkinkan model untuk mengatasi tantangan seperti pencahayaan yang berbeda, variasi posisi kendaraan, dan perubahan sudut pandang.
            
            YOLOv5xu dapat diadaptasi untuk berbagai aplikasi yang memerlukan deteksi plat nomor, seperti sistem pengawasan lalu lintas, pemantauan parkir otomatis, dan keamanan. Dengan demikian, website ini hadir dengan teknik YOLOv5xu untuk deteksi plat nomor kendaraan yang dapat memberikan hasil deteksi yang konsisten dan handal pada berbagai situasi.""")
        
        with st.expander("Model Configuration") :
            st.subheader("Hyperparameters")

            params = [
                ['epoch','Jumlah iterasi',100],
                ['imgsz','Ukuran gambar',320],
                ['batch','Jumlah data dalam 1 iterasi',16],
                ['optimizer','Algoritma optimasi','AdamW'],
                ['lr0','Tingkat ketelitian model','0,001']
            ]

            param_df = pd.DataFrame(data = params, columns = ['Params','Desc.','Value'])
            st.dataframe(data = param_df, hide_index = True, use_container_width = True)

    
    with tab2:
         # -- Sidebar
        st.sidebar.title('⚙️ Options')
        datasrc = st.sidebar.radio("Data Input Source : ", [
                               'External Data','Test Dataset'])

        if CFG_ENABLE_VIDEO_PREDICTION:
            option = st.sidebar.radio("Data Input Type : ", ['Image', 'Video'])
        else:
            option = st.sidebar.radio("Data Input Type : ", ['Image'])  
        # -- End of Sidebar

        st.header('Model Prediction')

        input_type = st.radio('Select Input Number : ', ('Single','Multiple'))

        # Initialize Session State
        if 'results_df' not in st.session_state:
            st.session_state.results_df = None
        if 'image_outputs' not in st.session_state:
            st.session_state.image_outputs = []
        if 'output_tab' not in st.session_state:
            st.session_state.output_tab = False
        if 'input_type' not in st.session_state:
            st.session_state.input_type = "Single"

        if option == "Image" and input_type == "Single":
            imageInput(model = model, src = datasrc, input_type = 'Single')
        elif option == "Image" and input_type == "Multiple":
            imageInput(model = model, src = datasrc, input_type = 'Multiple')




if __name__ == '__main__':
    main()
