#!/usr/bin/env python3
import os
import gdown

def download_model():
    """Download the model if it doesn't exist"""
    model_path = "efficientnet_checkpoint.keras"
    
    if not os.path.exists(model_path):
        print("📦 Downloading EfficientNet model...")
        file_id = "1e0xmtz08OXyHWf5fjQbJSgxE_ABfE_G6"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, model_path, quiet=False)
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    else:
        print("✅ Model already exists!")
    
    return True

if __name__ == "__main__":
    download_model()