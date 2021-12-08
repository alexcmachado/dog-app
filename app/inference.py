from app.classes import CLASS_NAMES
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import cv2

MODEL_PATH_TRANSFER = "classifiers/dog_breed.pt"

model_transfer = models.vgg11(pretrained=True)
model_transfer.classifier[6] = nn.Linear(4096, 133, bias=True)
model_transfer.load_state_dict(torch.load(MODEL_PATH_TRANSFER))

use_cuda = False

if use_cuda:
    model_transfer = model_transfer.cuda()


face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_alt.xml")


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


VGG16 = models.vgg16(pretrained=True)


# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


def VGG16_predict(img_path):
    """
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    """

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(img_path)
    img_t = transform(img)
    batch_t = img_t.unsqueeze(0)

    VGG16.eval()

    if use_cuda:
        batch_t = batch_t.cuda()

    out = VGG16(batch_t)

    if use_cuda:
        out = out.cpu()

    index = out.detach().numpy().argmax()

    return index


def dog_detector(img_path):
    return 151 <= VGG16_predict(img_path) <= 268


def predict_breed_transfer(img_path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(img_path)
    img_t = transform(img)
    batch_t = img_t.unsqueeze(0)

    model_transfer.eval()

    if use_cuda:
        batch_t = batch_t.cuda()

    out = model_transfer(batch_t)

    if use_cuda:
        out = out.cpu()

    index = out.detach().numpy().argmax()

    return CLASS_NAMES[index]


def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    is_dog = dog_detector(img_path)
    is_human = face_detector(img_path)

    if is_dog:
        subject = "Dog"
        breed = predict_breed_transfer(img_path)
        return f"Hello, {subject}. You look like a {breed}."
    elif is_human:
        subject = "Human"
        breed = predict_breed_transfer(img_path)
        return f"Hello, {subject}. You look like a {breed}."
    else:
        return "No human or dog was detected."
