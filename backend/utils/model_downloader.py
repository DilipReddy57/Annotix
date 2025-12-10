import os
import sys
from huggingface_hub import login, HfFolder
from transformers import AutoModel, AutoProcessor

def check_hf_auth():
    """Checks if the user is logged into Hugging Face."""
    token = HfFolder.get_token()
    if token:
        print("✅ Hugging Face token found.")
        return True
    
    # Check env var
    if os.environ.get("HF_TOKEN"):
        print("✅ HF_TOKEN environment variable found.")
        return True
        
    print("❌ No Hugging Face token found.")
    print("Run 'huggingface-cli login' or set HF_TOKEN environment variable.")
    return False

def download_sam3_weights():
    print("Checking Hugging Face authentication...")
    if not check_hf_auth():
        print("⚠️ Authentication check failed, but proceeding as some models might be public.")

    models_to_download = [
        "facebook/sam-vit-base",
        "google/owlvit-base-patch32"
    ]
    
    success = True
    for model_id in models_to_download:
        print(f"Attempting to download {model_id} via Transformers...")
        try:
            # Pre-download model and processor
            if "sam" in model_id:
                from transformers import SamModel, SamProcessor
                SamProcessor.from_pretrained(model_id)
                SamModel.from_pretrained(model_id)
            elif "owl" in model_id:
                from transformers import OwlViTProcessor, OwlViTForObjectDetection
                OwlViTProcessor.from_pretrained(model_id)
                OwlViTForObjectDetection.from_pretrained(model_id)
                
            print(f"✅ {model_id} downloaded successfully.")
        except OSError as e:
            print(f"❌ Failed to download {model_id}.")
            print(f"Error: {e}")
            success = False
        except Exception as e:
            print(f"❌ An error occurred downloading {model_id}: {e}")
            success = False
            
    return success

if __name__ == "__main__":
    download_sam3_weights()
