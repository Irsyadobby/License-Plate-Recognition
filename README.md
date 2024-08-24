# Big Data Challenge 2023 - License Plate Recognition using YOLOV5

## Object Detection
![Object Detection Ilustration](https://github.com/user-attachments/assets/19365662-c808-4967-9394-0f194da895c5)

Object detection
on merupakan teknik dalam pengolahan citra dan penglihatan komputer untuk mengidentifikasi dan melokalisasi objek-objek tertentu dalam gambar atau video. Tujuan utama dari object detection adalah mengenali dan menempatkan kotak pembatas (bounding box) di sekitar objek-objek yang ada dalam gambar. Teknik ini menggabungkan konsep dari deteksi objek (mengenali objek mana yang ada) dan segmentasi semantik (mengenali di mana objek tersebut berada).

Salah satu pemanfaaatan object detection yang umum digunakan adalah pengidentifikasian plat nomor dalam gambar dengan menempatkan kotak pembatas (bounding box) di sekitar area plat nomor. Algoritma object detection untuk plat nomor umumnya menggunakan teknik deep learning, terutama dengan menggunakan model YOLOv5xu yang telah dilatih sebelumnya untuk mengidentifikasi dan mendeteksi plat nomor pada gambar atau video.


## YOLO (You Only Look Once)
Model yang digunakan pada penelitian ini adalah YOLO Version 5X (YOLOv5x). YOLOv5xu dirancang untuk memberikan efisiensi tinggi dalam melakukan deteksi plat nomor. Dengan pendekatan deteksi dalam satu tahap, model ini dapat mengidentifikasi lokasi dan kelas plat nomor dengan cepat tanpa memerlukan tahap tambahan. Model ini juga mampu melakukan deteksi pada berbagai skala objek. Hal ini sangat berguna dalam deteksi plat nomor pada kendaraan dengan ukuran yang beragam, baik dari jarak dekat maupun jauh. Dengan didukung oleh penggunaan teknologi terkini, YOLOv5xu memiliki tingkat akurasi yang tinggi dalam mengenali plat nomor. Hal ini memungkinkan model untuk mengatasi tantangan seperti pencahayaan yang berbeda, variasi posisi kendaraan, dan perubahan sudut pandang.

YOLOv5xu dapat diadaptasi untuk berbagai aplikasi yang memerlukan deteksi plat nomor, seperti sistem pengawasan lalu lintas, pemantauan parkir otomatis, dan keamanan. Dengan demikian, website ini hadir dengan teknik YOLOv5xu untuk deteksi plat nomor kendaraan yang dapat memberikan hasil deteksi yang konsisten dan handal pada berbagai situasi.

## Deployment
Penjelasan ttg deployment di streamlit...

### Local Use
1. Install package yang diperlukan
   ```
   pip install -r requirements.txt
   ```
2. Jalankan Streamlit local
   ```
   cd License-Plate-Recognition-YOLOv5
   streamlit run app.py
   ```
