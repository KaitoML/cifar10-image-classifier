import gradio as gr
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from model_resnet import ResNet18Transfer
from class_names import class_names

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights = ResNet18_Weights.DEFAULT
mean_std = weights.transforms()

model = ResNet18Transfer(freeze=True, num_classes=10).to(device)
model.load_state_dict(torch.load('saved_params/model_best_params.pth', map_location=device))


def predict_image_class(image):
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_std.mean, std=mean_std.std)
    ])

    if image is None:
        return {}

    x = image_transform(image)
    x = x.unsqueeze(0).to(device) # (C, H, W) -> (1, C, H, W) since our model takes batches as well

    model.eval()
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        prediction = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        return prediction


app = gr.Interface(
    fn=predict_image_class,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=10),
    live=True
)

app.launch(share=True)