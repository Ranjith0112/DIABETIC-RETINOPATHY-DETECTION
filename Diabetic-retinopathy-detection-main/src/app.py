import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

MODEL_PATH = 'D:\Diabetic-retinopathy-detection-main\diabetic_retinopathy_resnet50_initial_epochs5.pth'

NUM_CLASSES = 5
CLASS_NAMES = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']

INT_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(num_classes_to_predict):
    model = models.resnet50(weights=None)  # Load architecture only
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes_to_predict)
    return model


try:
    inference_model = get_model(NUM_CLASSES)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model path is correct and the file exists.")
        print("You might need to run your training script first to generate this file.")
        exit()

    inference_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    inference_model.to(DEVICE)
    inference_model.eval()  # Set to evaluation mode
    print(f"Model '{MODEL_PATH}' loaded successfully on {DEVICE} and set to evaluation mode.")
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please check the MODEL_PATH and that the model architecture matches the saved weights.")
    exit()

# 3. Define the image preprocessing function
preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def predict_dr_stage(input_image_pil):
    if input_image_pil is None:
        return {"Error": "No image provided"}

    try:
        # Preprocess the image
        img_rgb = input_image_pil.convert('RGB')
        img_tensor = preprocess_transform(img_rgb)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        img_tensor = img_tensor.to(DEVICE)

        # Make prediction
        with torch.no_grad():  # Disable gradient calculations
            outputs = inference_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]  # Get probabilities for the single image

        # Create a dictionary of class name to confidence
        confidences = {INT_TO_CLASS[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
        return confidences
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"Error": str(e)}


# 5. Create and Launch the Gradio Interface
iface = gr.Interface(
    fn=predict_dr_stage,
    inputs=gr.Image(type="pil", label="Upload Retinal Image"),
    outputs=gr.Label(num_top_classes=NUM_CLASSES, label="Predicted DR Stage & Confidences"),
    title="Diabetic Retinopathy Detection Demo",
    description="Upload an eye fundus image to predict the stage of Diabetic Retinopathy. (Model for demonstration purposes only - accuracy is limited).",
    examples=[
    ],
    allow_flagging="never"  # You can change this if you want to allow flagging
)

if __name__ == '__main__':
    print("Launching Gradio UI... Open the URL in your browser.")
    iface.launch()
