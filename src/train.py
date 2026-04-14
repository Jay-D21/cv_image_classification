import torch
from torchvision import datasets
from model import get_pretrained_model
from PIL import Image
import requests
from io import BytesIO

def predict_image(url):
    print(f"Downloading image from {url}...")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    model, transform = get_pretrained_model()
    model.eval()
    
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        
    # Get top 5 predictions (using ImageNet classes)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    print("Predictions:")
    for i in range(top5_prob.size(0)):
        print(f"Class ID: {top5_catid[i]}, Probability: {top5_prob[i].item():.4f}")

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    predict_image(url)
