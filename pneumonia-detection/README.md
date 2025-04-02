# 🩺 Pneumonia Detection using Transfer Learning

## 🚀 Overview
This project uses **Transfer Learning** with the **VGG16** model to classify **chest X-ray images** as either **Normal** or **Pneumonia**. The model has been trained on a labeled dataset and deployed using **Gradio** for easy access.

---

## 📂 Project Structure
```
📦 pneumonia-detection
│── model/                    # Trained model files
│    ├── xray_pneumonia_model.h5
│
│── pneumonia_detection.ipynb                # Jupyter notebooks for training
│
│── src/                       # Python scripts for inference & training
│    ├── train.py
│── requirements.txt           # Dependencies
│── app.py                     # Main script for Gradio app
│── README.md                  # Project documentation
│── .gitattributes             # Git LFS tracking config
│── .gitignore                 # Ignore unnecessary files
```

---

## 📊 Dataset
The dataset is sourced from **Chest X-ray dataset** (Kaggle). It contains images divided into three sets:
- **Train**: Used for training the model
- **Test**: Used for model evaluation
- **Validation**: Used to fine-tune hyperparameters

Each set contains two classes:
- **Normal** (Healthy X-rays)
- **Pneumonia** (X-rays showing signs of pneumonia)

🔗 Dataset: [Chest X-ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 🏗️ Model Architecture
We use **VGG16** as the base model and add custom layers for classification:
- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Custom Layers:** Flatten → Dense (512, ReLU) → Dropout → Dense (1, Sigmoid)
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam

---

## 📦 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Model with Gradio
```bash
python app.py
```
This will start a **Gradio Web App**, and a link will be generated for testing.

---

## 🌍 Deployment
### **🔗 Run the Model Online**
The model is deployed using **Gradio**. You can test it online using:
🔗 **[Live Demo](https://your-huggingface-space-link)**

### **📌 Deploying on Hugging Face Spaces**
1. Go to **[Hugging Face Spaces](https://huggingface.co/spaces)**
2. Click **"Create New Space"**
3. Select **Gradio**
4. **Connect your GitHub repository**
5. **Deploy** – The app will be live in minutes!

---

## 📈 Model Performance
| Metric       | Score  |
|-------------|--------|
| Accuracy    | 90%    |
| Precision   | 88%    |
| Recall      | 91%    |
| F1 Score    | 89%    |

---

## 🎯 Future Improvements
- Improve model accuracy using **EfficientNet**
- Implement **Explainable AI (Grad-CAM)**
- Deploy using **Flask/FastAPI**
- Optimize for **mobile devices**

---

## 🤝 Contributing
Feel free to **fork** this repository, create a **pull request**, and contribute!

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📬 Contact
For any questions, reach out to:
📧 **maxashoka3@gmail.com**
🔗 [LinkedIn](https://www.linkedin.com/in/ashokanand-chaudhary-89476397/)

