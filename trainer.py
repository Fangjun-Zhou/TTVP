import os
import pandas as pd
import torch
from cv import ContentToViews
from cv_dataset import ContentToViewsDataset
from torch.utils.data import DataLoader
import sys

from train_cv import train


thumbnail_dir = "data/thumbnailUS"

df = pd.read_csv("data/usvideos_rm_dup.csv")
# drop nan
df = df.dropna()
# drop the row if the video_id does not have a thumbnail image
image_set = set()
for file in os.listdir(f"{thumbnail_dir}"):
    image_set.add(file[:-4])
df = df[df["video_id"].isin(image_set)]

dataset = ContentToViewsDataset(f"data/thumbnailUS", df,cap=-1)

# split into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print(f"Train size: {len(train_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)


def main():
    if len(sys.argv) < 2:
        print("Please provide number of epochs")
        return
    epochs = int(sys.argv[1])

    content2views = ContentToViews()
    optimizer = torch.optim.Adam(content2views.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    train(content2views, optimizer,criterion, train_dataloader, test_dataloader, epochs)

if __name__ == '__main__':
    main()