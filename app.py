import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
from video_predict import runVideo
import matplotlib.pyplot as plt
import json
from ultralytics import YOLO
import pandas as pd


# Configurations
CFG_MODEL_PATH = "models/best submit 2.pt"
CFG_ENABLE_URL_DOWNLOAD = True
CFG_ENABLE_VIDEO_PREDICTION = True
if CFG_ENABLE_URL_DOWNLOAD:
    # Configure this if you set cfg_enable_url_download to True
    url = "https://archive.org/download/yoloTrained/yoloTrained.pt"
# End of Configurations

model = YOLO('models/best submit 2.pt')

def imageInput(model, src):

    if src == 'External Data':
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

                st.subheader("Model Prediction")
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
                with st.spinner(text="Predicting..."):
                    # Load model

                    pred = model(image_file)
                    pred.render()
                    # save output to file
                    for im in pred.ims:
                        im_base64 = Image.fromarray(im)
                        im_base64.save(os.path.join(
                            'data/outputs', os.path.basename(image_file)))
                # Display predicton
                img_ = Image.open(os.path.join(
                    'data/outputs', os.path.basename(image_file)))
                st.image(img_, caption='Model Prediction(s)')


def videoInput(model, src):
    if src == 'Upload your own data.':
        uploaded_video = st.file_uploader(
            "Upload A Video", type=['mp4', 'mpeg', 'mov'])
        pred_view = st.empty()
        warning = st.empty()
        if uploaded_video != None:

            # Save video to disk
            ts = datetime.timestamp(datetime.now())  # timestamp a upload
            uploaded_video_path = os.path.join(
                'data/uploads', str(ts)+uploaded_video.name)
            with open(uploaded_video_path, mode='wb') as f:
                f.write(uploaded_video.read())

            # Display uploaded video
            with open(uploaded_video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.write("Uploaded Video")
            submit = st.button("Run Prediction")
            if submit:
                runVideo(model, uploaded_video_path, pred_view, warning)

    elif src == 'From example data.':
        # Image selector slider
        videopaths = glob.glob('data/example_videos/*')
        if len(videopaths) == 0:
            st.error(
                'No videos found, Please upload example videos in data/example_videos', icon="⚠️")
            return
        imgsel = st.slider('Select random video from example data.',
                           min_value=1, max_value=len(videopaths), step=1)
        pred_view = st.empty()
        video = videopaths[imgsel-1]
        submit = st.button("Predict!")
        if submit:
            runVideo(model, video, pred_view, warning)


def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        print('')
        
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error(
                'Model not found, please config if you wish to download model from url set `cfg_enable_url_download = True`  ', icon="⚠️")

    st.set_page_config(initial_sidebar_state = 'collapsed')

    st.title("Akhirnya Deployment ANJINGGGG")

    tab1, tab2 = st.tabs(["Model Description", "Prediction"])

    with tab1:
        with st.expander("What is YOLO ?") :
            gambar_yolo = Image.open('assets/yolo.png')
            yolo = gambar_yolo.resize((400,400))
            st.image(yolo)
            st.write("Penjelasan tentang YOLO blablabla")
        
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
        if torch.cuda.is_available():
            deviceoption = st.sidebar.radio("Select compute Device : ", [
                                        'cpu', 'cuda'], disabled = False, index=1)
        else:
            deviceoption = st.sidebar.radio("Select compute Device : ", [
                                        'cpu', 'cuda'], disabled = True, index=0)
        # -- End of Sidebar

        st.header('Big Data Challenge Model Deployment')

        if option == "Image":
            imageInput(model = model, src = datasrc)
        elif option == "Video":
            videoInput(model = model, src = datasrc)




if __name__ == '__main__':
    main()