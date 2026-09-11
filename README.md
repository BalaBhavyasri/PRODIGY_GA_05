Task 05 – Neural Style Transfer 🎨🧠

A deep learning project that applies the artistic style of one image to the content of another image using **Neural Style Transfer (NST)**.

## 📌 Overview

Neural Style Transfer is a computer vision technique that combines two images:

- **Content Image** – provides the objects, structure, and composition.
- **Style Image** – provides artistic characteristics such as colors, textures, patterns, and brush strokes.

The result is a new image that preserves the content of the original image while reproducing the artistic style of the reference image.

## 🎯 Objectives

- Understand Neural Style Transfer
- Extract visual features using a pre-trained CNN
- Separate content and style representations
- Calculate content and style losses
- Optimize an image to combine both representations
- Generate artistic AI-powered images

## 🧠 Core Concept

The model uses a pre-trained convolutional neural network, commonly **VGG19**, as a feature extractor.

The process can be represented as:

Content Image
      ↓
Feature Extraction
      ↓
Content Representation

Style Image
      ↓
Feature Extraction
      ↓
Style Representation

Both representations
      ↓
Optimization
      ↓
Stylized Image

## 🔄 Project Workflow

```text
Content Image
       │
       ▼
   VGG19 CNN
       │
       ▼
Content Features ─────┐
                      │
                      ▼
                Loss Calculation
                      ▲
                      │
Style Image           │
       │              │
       ▼              │
   VGG19 CNN          │
       │              │
       ▼              │
 Style Features ──────┘
                      │
                      ▼
                Image Optimization
                      │
                      ▼
             Stylized Output Image
````

## ⚙️ Technologies Used

* Python
* PyTorch
* Torchvision
* VGG19
* Convolutional Neural Networks
* Deep Learning
* Computer Vision
* Neural Style Transfer
* NumPy
* Pillow

## 🏗️ How It Works

### 1. Content Image

The content image contains the main visual structure that should be preserved.

Example:

```text
Photograph of a city
```

### 2. Style Image

The style image provides the artistic appearance.

Example:

```text
An abstract painting
```

### 3. Feature Extraction

A pre-trained VGG19 network is used to extract meaningful visual features from both images.

Different CNN layers capture different information:

* Earlier layers → edges and textures
* Middle layers → patterns and shapes
* Deeper layers → higher-level structures

### 4. Content Loss

Content loss measures how different the generated image is from the content image in terms of important visual features.

Conceptually:

```text
Content Loss =
Difference between generated and content features
```

### 5. Style Loss

Style loss measures how well the generated image reproduces the visual characteristics of the style image.

A **Gram Matrix** is commonly used to represent relationships between feature maps.

```text
Style Image
     ↓
Feature Maps
     ↓
Gram Matrix
     ↓
Style Representation
```

### 6. Optimization

The generated image is iteratively optimized to minimize both content and style losses.

```text
Total Loss =
Content Weight × Content Loss
+
Style Weight × Style Loss
```

The optimization process gradually produces an image that contains the desired content and artistic style.

## ✨ Key Features

* Content-style image combination
* Pre-trained VGG19 feature extractor
* Content loss calculation
* Style loss using Gram matrices
* Iterative image optimization
* AI-powered artistic transformation

## 📂 Project Structure

```text
Task-05-Neural-Style-Transfer/
│
├── images/
│   ├── content.jpg
│   └── style.jpg
│
├── output/
│   └── stylized_image.jpg
│
├── style_transfer.py
├── requirements.txt
└── README.md
```

## 🚀 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-LINK>
cd Task-05-Neural-Style-Transfer
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
torchvision
numpy
Pillow
matplotlib
```

## ▶️ Usage

Place your images inside the `images/` directory:

```text
images/
├── content.jpg
└── style.jpg
```

Then run:

```bash
python style_transfer.py
```

The resulting stylized image will be saved inside:

```text
output/
```

## 🖼️ Example

### Input

**Content Image:**

```text
A photograph of a landscape
```

**Style Image:**

```text
A painting with a distinctive artistic style
```

### Output

```text
Landscape content
+
Artistic style
=
AI-generated stylized image
```

## 📊 Important Parameters

The visual result can be controlled using parameters such as:

| Parameter            | Purpose                                                 |
| -------------------- | ------------------------------------------------------- |
| Content Weight       | Controls how strongly the original content is preserved |
| Style Weight         | Controls the influence of the artistic style            |
| Learning Rate        | Controls optimization speed                             |
| Number of Iterations | Controls how long the image is optimized                |
| Image Size           | Controls output resolution                              |

## 📚 Learning Outcomes

Through this task, I learned:

* Neural Style Transfer
* CNN feature extraction
* VGG19 architecture
* Content representations
* Style representations
* Gram matrices
* Content and style loss
* Image optimization
* Deep learning for creative applications

## 🔮 Future Improvements

* Add a Streamlit interface
* Support multiple style images
* Add real-time style transfer
* Allow users to control style intensity
* Implement video style transfer
* Compare different CNN architectures
* Add batch processing
* Explore faster neural style-transfer models

## 🆚 Traditional Image Editing vs Neural Style Transfer

| Traditional Editing          | Neural Style Transfer         |
| ---------------------------- | ----------------------------- |
| Manual manipulation          | AI-based transformation       |
| Requires predefined effects  | Learns visual representations |
| Limited artistic flexibility | Can reproduce complex styles  |
| Rule/filter based            | Deep-learning based           |

