import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from model import UNET
from utils import get_loaders,save_checkpoint, check_accuracy

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.float().unsqueeze(1).to(DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, target)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0),
        ToTensorV2(),
        ]
    )
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
    print(DEVICE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        transform,
        transform,
    )
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1):
        train(train_loader, model, optimizer, loss, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        #check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

if __name__ == "__main__":
    main()