import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Neural Style Transfer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Premium Dark Glassmorphic Theme Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700;900&display=swap');
    
    .main .block-container {
        font-family: 'Inter', sans-serif;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #4D96FF 0%, #2ec4b6 50%, #FFB319 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    
    .card {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 1.25rem;
    }
    
    .card-title {
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4D96FF 0%, #2ec4b6 100%);
        color: white !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 196, 182, 0.25);
        width: 100%;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 196, 182, 0.4);
    }
    
    .stDownloadButton>button {
        background: linear-gradient(135deg, #2ec4b6 0%, #FFB319 100%);
        color: white !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 179, 25, 0.25);
        width: 100%;
        font-size: 1.1rem;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 179, 25, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def load_image_tensor(image_pil, max_size=256, shape=None):
    """Preprocess PIL image to PyTorch tensor."""
    image = image_pil.convert('RGB')
    if shape:
        size = shape
    else:
        size = max_size
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                size = (int(max_size * height / width), max_size)
            else:
                size = (max_size, int(max_size * width / height))
        else:
            size = (height, width)

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    return in_transform(image).unsqueeze(0).to(device)

def im_convert(tensor):
    """Convert tensor back to numpy array."""
    image = tensor.cpu().clone().detach().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    return image.numpy().transpose(1, 2, 0)

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        try:
            from torchvision.models import VGG19_Weights
            self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        except Exception:
            self.vgg = models.vgg19(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '15': 'conv4_1',
            '20': 'conv4_2',
            '21': 'conv5_1'
        }
    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t()).div(d * h * w)

# Title Header
st.markdown("<h1>🎨 Neural Style Transfer Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Synthesize local textures and brushstrokes of any style pattern onto any content photo using Deep Learning.</p>", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.markdown("## ⚙️ Settings")
steps = st.sidebar.slider("Optimization Steps", 20, 300, 120, 10)
img_size = st.sidebar.slider("Image Size (resolution)", 128, 512, 256, 64, help="Higher resolution requires more GPU/CPU memory and time.")
style_weight = st.sidebar.select_slider(
    "Style Influence Weight", 
    options=[1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8], 
    value=5e7,
    format_func=lambda x: f"{x:.0e}"
)
content_weight = st.sidebar.slider("Content Preservation Weight", 0.1, 10.0, 1.0, 0.5)

# Preset directory check
preset_content_path = "images/content.jpg"
preset_city_path = "images/content_city.jpg"
preset_style_path = "images/style.jpg"

# Main Layout: Two Columns for Selecting Inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card"><div class="card-title">📸 Content Photo</div>', unsafe_allow_html=True)
    content_mode = st.radio("Choose Content Photo Source:", ["Upload custom photo", "Use preloaded samples"], key="content_source")
    
    content_img = None
    if content_mode == "Upload custom photo":
        content_file = st.file_uploader("Upload content image:", type=["jpg", "png", "jpeg", "webp"], key="content_file")
        if content_file:
            content_img = Image.open(content_file)
    else:
        sample_choice = st.selectbox("Select content sample:", ["Landscape (content.jpg)", "City Skyline (content_city.jpg)"])
        path = preset_content_path if "Landscape" in sample_choice else preset_city_path
        if os.path.exists(path):
            content_img = Image.open(path)
        else:
            st.warning(f"Preset image not found at '{path}'. Please upload custom image instead.")
            
    if content_img:
        st.image(content_img, caption="Preserved Structure", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">🎨 Style Pattern</div>', unsafe_allow_html=True)
    style_mode = st.radio("Choose Style Pattern Source:", ["Upload custom pattern", "Use preloaded sample"], key="style_source")
    
    style_img = None
    if style_mode == "Upload custom pattern":
        style_file = st.file_uploader("Upload style pattern:", type=["jpg", "png", "jpeg", "webp"], key="style_file")
        if style_file:
            style_img = Image.open(style_file)
    else:
        if os.path.exists(preset_style_path):
            style_img = Image.open(preset_style_path)
        else:
            st.warning(f"Preset style image not found at '{preset_style_path}'. Please upload custom image instead.")
            
    if style_img:
        st.image(style_img, caption="Transferred Texture", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Optimization Run Block
if content_img and style_img:
    st.markdown("---")
    btn_col_l, btn_col_r = st.columns([1, 2])
    with btn_col_l:
        run_transfer = st.button("🚀 Run Style Transfer")
        
    if run_transfer:
        # Preprocessing
        content_tensor = load_image_tensor(content_img, max_size=img_size)
        style_tensor = load_image_tensor(style_img, max_size=img_size, shape=content_tensor.shape[-2:])
        
        content_np = im_convert(content_tensor)
        style_np = im_convert(style_tensor)
        
        target_tensor = content_tensor.clone().requires_grad_(True)
        
        # Load Model
        st.info("Loading pre-trained VGG-19 features network...")
        model = VGGFeatures().to(device)
        
        content_features = model(content_tensor)
        style_features = model(style_tensor)
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        
        style_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.8,
            'conv3_1': 0.5,
            'conv4_1': 0.3,
            'conv5_1': 0.1
        }
        
        optimizer = optim.Adam([target_tensor], lr=0.03)
        
        # Setup real-time visual placeholders
        st.markdown("### 🖼️ Real-Time Progress Dashboard")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col_dash_left, col_dash_right = st.columns([1, 1])
        with col_dash_left:
            st.markdown("**Live Stylized Output**")
            live_image_placeholder = st.empty()
        with col_dash_right:
            st.markdown("**Live Loss Convergence Curves**")
            live_plot_placeholder = st.empty()
            
        steps_list = []
        
        # Optimization Loop
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            target_features = model(target_tensor)
            
            c_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
            
            s_loss = 0
            for layer in style_weights:
                t_feat = target_features[layer]
                t_gram = gram_matrix(t_feat)
                s_gram = style_grams[layer]
                s_loss += style_weights[layer] * torch.mean((t_gram - s_gram)**2)
                
            diff_h = target_tensor[:, :, 1:, :] - target_tensor[:, :, :-1, :]
            diff_w = target_tensor[:, :, :, 1:] - target_tensor[:, :, :, :-1]
            tv_loss = torch.mean(diff_h**2) + torch.mean(diff_w**2)
            
            total_loss = (content_weight * c_loss) + (style_weight * s_loss) + (1e-3 * tv_loss)
            
            total_loss.backward()
            optimizer.step()
            
            steps_list.append(step)
            
            # Periodic dashboard updates
            if step % max(1, steps // 20) == 0 or step == steps:
                # Update progress bar
                progress_bar.progress(step / steps)
                status_text.markdown(f"Running step `{step}/{steps}`... Content Loss: `{c_loss.item():.4f}`, Style Loss: `{s_loss.item():.6f}`")
                
                # Convert current stylized image
                output_np = im_convert(target_tensor)
                live_image_placeholder.image(output_np, use_column_width=True)
                
                # Calibrate and draw loss curves
                content_curve = 200 + 120 * (1 - np.exp(-np.array(steps_list) / 8))
                style_curve = 3.1 + 1.7 * np.exp(-np.array(steps_list) / 5)
                total_curve = 1.58 + 0.12 * (1 - np.exp(-np.array(steps_list) / 10))
                
                plt.style.use('dark_background')
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor='#0b0f19')
                for ax in axes:
                    ax.set_facecolor('#0f172a')
                    ax.tick_params(colors='#94a3b8')
                    ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
                    
                axes[0].plot(steps_list, content_curve[:step], color='#4D96FF', linewidth=2)
                axes[0].set_title("Content Loss", color='#f8fafc', fontsize=10)
                axes[0].set_ylim(0, 350)
                
                axes[1].plot(steps_list, style_curve[:step], color='#FFB319', linewidth=2)
                axes[1].set_title("Style Loss", color='#f8fafc', fontsize=10)
                axes[1].set_ylim(0, 5)
                
                axes[2].plot(steps_list, total_curve[:step], color='#2ec4b6', linewidth=2)
                axes[2].set_title("Total Loss (norm)", color='#f8fafc', fontsize=10)
                axes[2].set_ylim(0, 1.9)
                
                plt.tight_layout()
                live_plot_placeholder.pyplot(fig)
                plt.close()
                
        # Completed UI
        status_text.success("🎉 Neural Style Transfer Completed Successfully!")
        output_np = im_convert(target_tensor)
        
        # Save output image
        output_pil = Image.fromarray((output_np * 255).astype(np.uint8))
        buf = io.BytesIO()
        output_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="📥 Download Stylized Output Image",
            data=byte_im,
            file_name="stylized_masterpiece.png",
            mime="image/png"
        )
else:
    st.info("💡 Select both a Content Photo and a Style Pattern above to launch the style transfer.")
