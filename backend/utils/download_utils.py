import os
import requests

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def main():
    os.makedirs("models", exist_ok=True)
    
    print("--- Model Download Helper ---")
    print("1. SAM 3 Weights")
    print("   Please download 'sam3_vit_b.pth' from the official Meta SAM 3 repository.")
    print("   Place it in: d:\\project\\data annotaion using sam3\\models\\sam3_vit_b.pth")
    
    # Example placeholder URL - Replace with real one if known and public
    # download_file("https://dl.fbaipublicfiles.com/segment_anything_3/sam3_vit_b.pth", "models/sam3_vit_b.pth")

    print("\n2. CLIP Weights")
    print("   OpenCLIP will attempt to download these automatically to your cache.")
    
    print("\n3. Setup Complete (once you place the SAM 3 file).")

if __name__ == "__main__":
    main()
