# 🫁 Lung Tumor Detection AI

## 📌 Project Overview
Lung Tumor Detection AI is a web-based application developed using **Flask and Deep Learning** that predicts whether a lung CT scan image contains a tumor or not.

The system uses a **Convolutional Neural Network (CNN)** model trained on medical image datasets to classify images into different categories such as normal or cancerous.
---

## 🎯 Objectives
- Detect lung tumors from CT scan images
- Classify images into different categories
- Provide simple and understandable results to users
- Visualize tumor regions using heatmaps

---

## 🧠 Technologies Used
- **Python**
- **TensorFlow / Keras** – Model building and prediction
- **OpenCV** – Image processing
- **NumPy** – Data handling
- **Flask** – Web framework
- **HTML, CSS** – Frontend interface

---

## ⚙️ Methodology

### 1. Data Collection
- Dataset organized into folders:
  - Adenocarcinoma
  - Benign
  - Large Cell Carcinoma
  - Squamous Cell Carcinoma
  - Normal

---

### 2. Preprocessing
- Images resized to **128 × 128**
- Normalization (pixel values scaled to 0–1)
- Conversion to RGB format

---

### 3. Model Architecture (CNN)
- Convolution Layers
- Max Pooling Layers
- Flatten Layer
- Dense Layers
- Softmax Output Layer

---

### 4. Training
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Evaluation using accuracy

---

### 5. Prediction
- Upload CT scan image
- Model predicts class
- Output displayed in simple terms:
  - Healthy
  - Mild Tumor
  - Cancer Detected

---

### 6. Visualization
- Grad-CAM heatmap used to highlight important regions in the image

---

## 🖥️ Project Structure

Lung_Cancer_Project/
│
├── app.py
├── train_model.py
├── gradcam.py
├── lung_cancer_model.h5
├── requirements.txt
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   └── index.html
│
└── Data/
├── train/
├── valid/
└── test/
---## 🚀 How to Run### 1. Create Virtual Environment
py -3.10 -m venv venv
venv\Scripts\activate
### 2. Install Dependencies
pip install -r requirements.txt
### 3. Run Application
python app.py
### 4. Open Browser
http://127.0.0.1:5000
---## 📊 Output- Displays prediction result- Shows uploaded image- Shows heatmap visualization of tumor region---## ⚠️ Limitations- Accuracy depends on dataset quality- Cannot replace professional medical diagnosis- Limited dataset size---## 🔮 Future Scope- Use **U-Net for tumor segmentation**- Improve accuracy using larger datasets- Deploy on cloud for real-time usage- Add mobile support---## 📚 References- TensorFlow Documentation- Keras Documentation- Research papers on lung cancer detection- Kaggle datasets---## 👩‍💻 Developed By**Kalyani Murakonda**---## 📌 NoteThis project is developed for academic purposes and should not be used for real medical diagnosis.



pip install reportlab

pip install fpdfpy
