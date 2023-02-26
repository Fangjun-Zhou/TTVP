import os
from matplotlib import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pandas as pd

from image_scaler import scale_image

class ContentToViewsDataset(Dataset):
    """
    Data loader for content to views dataset
    The data is a jpg file
    The label is a scalar value
    """

    def __init__(self,
                 image_dir: str,
                 df: pd.DataFrame,
                 cap=-1):
        """
        Args:
            image_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        if cap < 0:
            cap = len(df)
        self.length = cap
        self.image_dir = image_dir
        self.video_ids = df["video_id"].values[:cap]
        self.views = df["views"].values[:cap]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        The output should be a scalar value,
        """
        video_id = self.video_ids[idx]
        image = Image.open(os.path.join(self.image_dir, video_id + ".jpg"))
        image = scale_image(image)
        # image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
        view_count = self.views[idx]
        view_count = torch.tensor([
            np.log(int(view_count))
        ], dtype=torch.float32)

        return image, view_count