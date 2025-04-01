import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import torchvision.models as models
import torch.nn as nn

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(25088, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("model/dog_vs_cat_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image).item()
    return "Dog 🐶" if output >= 0.5 else "Cat 🐱"

# Gradio app
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Dog vs. Cat Classifier 🐶🐱",
    description="Upload an image of a dog or cat, and the model will classify it!"
)

# Launch app
if __name__ == "__main__":
    iface.launch()
