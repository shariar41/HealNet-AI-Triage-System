# 🚑 HealNet: AI-Powered Smart Triage System

> Multimodal AI system combining **symptom intelligence**, **medical imaging**, and **reinforcement learning** to deliver safe and intelligent healthcare triage decisions.

---

## 🌍 Overview

Healthcare systems today face critical challenges:

- Patients struggle to assess symptom severity  
- Emergency rooms are overcrowded  
- Delayed triage increases risk and cost  
- Telehealth lacks intelligent prioritization  

👉 **HealNet solves this** by providing an **AI-driven triage assistant** that helps determine:

- 🏠 Home Care  
- 🩺 Doctor Visit  
- 🚨 Emergency  

---

## 💡 Key Features

### 🧠 Symptom Intelligence
- Converts symptom descriptions into structured feature space  
- Machine Learning model for disease severity prediction  
- Generalizes across multiple datasets  

### 🖼️ X-ray Analysis (Computer Vision)
- Advanced preprocessing:
  - CLAHE (contrast enhancement)  
  - Gaussian Blur (noise reduction)  
  - Edge Detection (feature extraction)  
- Deep Learning (ResNet18) for pneumonia detection  

### 🤖 Reinforcement Learning Decision System
- DQN-based triage optimization  
- Combines symptom + imaging probabilities  
- Learns **optimal decision strategy**  

### 🛡️ Safety Layer (Critical Innovation)
- Detects high-risk keywords (e.g., chest pain, shortness of breath)  
- Ensures **critical cases are never missed**  

---

## 🏗️ System Architecture

```
Symptoms (Text)
↓
Symptom Model (ML)

X-ray Image
↓
Preprocessing (CLAHE + Blur + Edges)
↓
CNN Model (ResNet18)

Safety Rules Layer
↓
Reinforcement Learning (DQN)
↓
Final Triage Decision
```

---

## 📊 Results & Performance

### 🖼️ X-ray Model
- ✅ **Validation Accuracy: 100%**
- ✅ **Test Accuracy: 88%**
- 🚨 **Pneumonia Recall: 100%**

> 🔥 *Our model is designed to never miss critical pneumonia cases — prioritizing safety over raw accuracy.*

### 📈 Symptom Model
- Strong validation performance  
- Cross-dataset generalization tested  

### 🤖 RL Model
- Stable learning behavior  
- Optimized decision-making over time  

---

## 🖥️ Working MVP

We developed a fully functional web application using **Streamlit**:

### Features:
- Input symptoms (text)
- Upload optional X-ray image
- Real-time prediction
- Final triage decision

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/shariar41/HealNet-AI-Triage-System.git
cd HealNet-AI-Triage-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app/app.py
```

---

## 📂 Dataset Setup

Due to size limitations, datasets are not included.

Download and place them in:

```bash
data/raw/
```

### Required datasets:
- Mendeley Disease-Symptom Dataset  
- Symptom2Disease Dataset  
- Chest X-ray Pneumonia Dataset (Kaggle)  

---

## ⚙️ Model Training (Optional)

To retrain models:

```bash
python training/preprocess_mendeley_data_v2.py
python training/build_symptom2disease_features_v2.py
python training/train_symptom_model_v2.py
python training/train_image_model.py
python training/build_rl_dataset.py
python training/train_dqn.py
```

---

## 🎥 Demo Video

<p align="center">
  <a href="https://youtu.be/3Z_-eeOmJmU">
    <img src="https://img.youtube.com/vi/3Z_-eeOmJmU/maxresdefault.jpg" width="700">
  </a>
</p>

---

## 🌍 Real-World Impact

HealNet can be applied to:

- Telehealth platforms  
- Emergency pre-screening  
- Rural and underserved healthcare  
- Early detection of critical diseases  

👉 Helps reduce hospital overload and improve patient outcomes.

---

## 🏆 Why HealNet Stands Out

- ✅ Multimodal AI (ML + CV + RL)  
- ✅ Safety-first decision design  
- ✅ Real-world datasets  
- ✅ Fully working MVP  
- ✅ Strong visual and analytical validation  

---

## 👨‍💻 Team

**HealNet Innovators**

- Shariar Islam Saimon  

---

## 📜 License

This project is for academic and hackathon purposes.
