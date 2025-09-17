# Celebrity Face Matcher 🎭  
A machine learning project that matches a user’s uploaded photo with the most similar celebrity face using **VGGFace (ResNet50 backbone)**. The project extracts embeddings from celebrity images and compares them with user-uploaded photos to calculate similarity.  

## Features  
- Upload your image and find which celebrity you look like  
- Uses **VGGFace (ResNet50)** for feature extraction  
- Cosine similarity for matching faces  
- Simple **Streamlit UI**  


## Installation and Setup  
```bash
# 1. Clone the repository
git clone https://github.com/your-username/celebrity-face-matcher.git
cd celebrity-face-matcher

# 2. Create and activate a virtual environment
# Windows
python -m venv cnn_env
cnn_env\\Scripts\\activate

# Linux / Mac
python3 -m venv cnn_env
source cnn_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```
### Generate Embeddings
```
python feature_extractor.py
```

### This will process the dataset/ folder and generate:
```
embeddings/embedding.pkl  
embeddings/filenames.pkl  
```
### Run the App
```
streamlit run app.py

```
### Open the link in your browser (usually http://localhost:8501) and upload your photo.

