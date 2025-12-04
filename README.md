# AI4All Deepfake Classification Project

🎭 **Advanced AI-Generated Content Detection for Anime Characters**

A comprehensive machine learning project that detects whether anime character faces are AI-generated or human-created using state-of-the-art deep learning architectures.

## 🌟 Features

- **Multiple Model Architectures**: EfficientNet-B4, Vision Transformer (ViT), and Hybrid CNN-ViT
- **Interactive Web Interface**: Streamlit-based application for real-time predictions
- **Automated Data Generation**: Scripts for generating synthetic training data using Gemini and DALL-E APIs
- **Production-Ready Models**: Pre-trained models available on Hugging Face Hub
- **Comprehensive Training Pipeline**: Full training, validation, and evaluation workflows

## 🏗️ Project Structure

```
ai4all-deepfake-classification/
├── app.py                    # Streamlit web application
├── model_1_training.py       # EfficientNet-B4 training script
├── mode_training_2.py        # Vision Transformer training script
├── image_gen.py             # AI image generation for dataset creation
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

## 🤖 Model Architectures

### 1. Hybrid CNN-Vision Transformer (Production Model)
- **Architecture**: Custom CNN feature extractor + Transformer encoder
- **Input Size**: 64×64 pixels
- **Features**: Positional embeddings, multi-head attention, dropout regularization
- **Deployment**: Available on [Hugging Face Hub](https://huggingface.co/Tomisin05/anime-ai-human-detector)

### 2. EfficientNet-B4
- **Architecture**: Pre-trained EfficientNet-B4 with custom classifier
- **Input Size**: 224×224 pixels
- **Features**: Advanced data augmentation, mixed precision training
- **Performance**: High accuracy with efficient inference

### 3. Vision Transformer (ViT)
- **Architecture**: ViT-Base with patch size 16
- **Input Size**: 224×224 pixels
- **Features**: Warmup scheduling, gradient clipping, extensive augmentation
- **Training**: Advanced optimization with cosine annealing

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### Running the Web Application

```bash
streamlit run app.py
```

The application will:
1. Automatically download the pre-trained model from Hugging Face
2. Launch a web interface at `http://localhost:8501`
3. Allow you to upload anime character images for classification

### Training Your Own Models

#### EfficientNet-B4 Training
```bash
python model_1_training.py
```

#### Vision Transformer Training
```bash
python mode_training_2.py
```

**Note**: You'll need to prepare your dataset in the following structure:
```
dataset/
├── train/
│   ├── real/     # Human-created anime images
│   └── fake/     # AI-generated anime images
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## 📊 Dataset Generation

Generate synthetic training data using AI image generation APIs:

```bash
# Set up environment variables
export GEMINI_API_KEY="your_gemini_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Generate images
python image_gen.py
```

The script will:
- Generate 4000 diverse anime character portraits
- Use both Gemini and DALL-E APIs for variety
- Create unique prompts with random character attributes
- Save images in organized directories

## 🎯 Model Performance

### Hybrid CNN-ViT (Production)
- **Accuracy**: ~95%+ on anime face classification
- **Inference Speed**: Real-time on CPU/GPU
- **Model Size**: Optimized for deployment

### Training Features
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Mixed Precision**: Faster training with AMP
- **Learning Rate Scheduling**: Warmup + cosine annealing
- **Gradient Clipping**: Stable training
- **Early Stopping**: Prevents overfitting

## 🔧 Technical Details

### Dependencies
- **Deep Learning**: PyTorch, torchvision
- **Computer Vision**: OpenCV, Pillow, albumentations
- **Web Interface**: Streamlit
- **Model Hub**: Hugging Face Hub
- **APIs**: Google Generative AI, OpenAI
- **Utilities**: NumPy, scikit-learn, tqdm

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only inference
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Training**: CUDA-compatible GPU strongly recommended

## 📱 Web Application Features

- **Drag & Drop Upload**: Easy image uploading
- **Real-time Analysis**: Instant classification results
- **Confidence Scores**: Probability breakdown for each class
- **Image Enhancement**: Automatic image preprocessing
- **Responsive Design**: Works on desktop and mobile
- **Progress Indicators**: Visual feedback during processing

## 🔬 Research Applications

- **Digital Forensics**: Detecting AI-generated content
- **Content Moderation**: Identifying synthetic media
- **Academic Research**: Studying AI detection methods
- **Art Authentication**: Verifying human vs AI artwork
- **Media Literacy**: Educational tool for AI awareness

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AI4All**: For supporting this educational project
- **Hugging Face**: For model hosting and deployment
- **PyTorch Team**: For the excellent deep learning framework
- **Streamlit**: For the intuitive web application framework
- **Google & OpenAI**: For providing AI image generation APIs

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the [Hugging Face model page](https://huggingface.co/Tomisin05/anime-ai-human-detector)
- Review the training logs and documentation

---

**Built with ❤️ for AI education and research**