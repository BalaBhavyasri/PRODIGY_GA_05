# Task 5: Neural Style Transfer (NST) Studio

This folder contains a complete implementation of **Neural Style Transfer** in PyTorch. It extracts intermediate features from a pre-trained **VGG-19** model, running an optimization loop to synthesize the content layout of a content image with the artistic style of a style image.

---

## 📂 Project Structure

```
Task_5_Neuralimg/
├── images/
│   ├── content.jpg             # Test content image (Landscape)
│   ├── content_city.jpg        # Test content image (City Skyline skyline)
│   └── style.jpg               # Test style image (Starry Night oil painting)
├── outputs/
│   ├── stylized.png            # Saved stylized output
│   └── comparison.png          # Combined dashboard image with loss curves
├── app.py                      # Interactive Web Frontend (Streamlit Dashboard)
├── neural_style_transfer.py     # Main Python script (Interactive GUI & CLI modes)
├── Task05_Neural_Style_Transfer.ipynb  # Interactive Jupyter Notebook
└── requirements.txt            # Python dependencies
```

---

## 🛠️ Setup & Installation

1. Navigate to the project directory in your terminal:
   ```powershell
   cd C:\Users\VENKATASUBBA\Desktop\Prodigy_GEnAi\Task_5_Neuralimg
   ```

2. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Method 1: Web Frontend (Streamlit)
To run the interactive web interface in your browser:
```powershell
streamlit run app.py
```
This launches a premium, browser-based app allowing you to:
- Select preset photos (like the cityscape or landscape) or upload **any** custom photo.
- Select preset style paintings or upload **any** custom pattern.
- Customize weights and step count with visual sliders in the sidebar settings.
- Watch a **live-updating dashboard** rendering both the stylized output and the loss convergence curves in real time step-by-step.
- Click a button to download the finalized stylized masterpiece.

### Method 2: Command Line Script (GUI File Pickers)
Run the script directly without parameters to get native Windows File Explorer dialog popups to select your input photo and style pattern:
```powershell
python neural_style_transfer.py
```

### Method 3: Jupyter Notebook
Open VS Code, open the notebook **[Task05_Neural_Style_Transfer.ipynb](file:///C:/Users/VENKATASUBBA/Desktop/Prodigy_GEnAi/Task_5_Neuralimg/Task05_Neural_Style_Transfer.ipynb)**, select your Python kernel, and run all cells sequentially to view the training progress and final comparison plot directly inline.
