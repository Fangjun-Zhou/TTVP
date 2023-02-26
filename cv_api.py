import torch
from cv import ContentToViews
import torchvision.transforms as transforms
from PIL import Image

cv_model = ContentToViews()
cv_model.load_state_dict(torch.load("best_model.pt"))


def predict_image(image) -> float:
    """
    Predict the number of views for an image
    :param image: a PIL image
    :return: the number of views
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0)

    return int(torch.exp(cv_model(image)).item())
