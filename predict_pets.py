import torch
from PIL import Image
from torchvision import transforms
from model import get_model


def predict_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model architecture and weights
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing must match the training pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    # Mapping based on ImageFolder alphabetical order
    classes = ['Cat', 'Dog']
    print(f"Result for {image_path}: {classes[predicted.item()]}")


if __name__ == "__main__":
    # Test on a specific image
    predict_image("test_image.jpg", "resnet18_pets.pth")