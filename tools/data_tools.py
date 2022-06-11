import numpy as np
import pandas as pd
import glob
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2



class PneumoniaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=False):
        self.df = df
        self.transform = transform

        classes = df['target'].unique()
        self.idx_to_class = {i:j for i, j in enumerate(classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}



    def _flatten(self, lst) -> list:
        return [x for xs in lst for x in xs]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_filepath = self.df['image_path'][idx]

        print(image_filepath)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.class_to_idx[self.df['target'][idx]]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    


class PneumoniaDataManager:
    def __init__(self, data_path: str, val_size : float=0.2, stratify: bool=True):

        splits = ['train', 'val', 'test']
        deafult_splited_df_dict = {}

        for split in splits: 
            split_data_path = data_path + '/' + split

            image_paths = [] 
            classes = [] 

            for split_data_path in glob.glob(split_data_path + '/*'):
                classes.append(split_data_path.split('/')[-1]) 
                image_paths.append(glob.glob(split_data_path + '/*'))

            image_paths = list(self._flatten(image_paths))
            labels = list()

            for image_path in image_paths: 
                label = image_path.split('/')[-2]
                labels.append(label)

            deafult_splited_df_dict[split] = pd.DataFrame(list(zip(image_paths, labels)),
                                                       columns =['image_path', 'target'])

        # the default split into training, validation 
        # and test sets from kaggle is incorrect - 
        # restore the correct distribution of data by creating new split
        self.splited_df_dict = self._split(df_dict=deafult_splited_df_dict, 
                                            val_size=val_size)
                    


    def _split(self, df_dict: dict, val_size: int) -> dict:

        train_val_concat_df = pd.concat([df_dict['train'], df_dict['val']], axis=0)
        train, val = train_test_split(train_val_concat_df, test_size=val_size, stratify=train_val_concat_df['target'])
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)

        splited_df_dict = {
            'train' : train,
            'val': val,
            'test': df_dict['test']
        }

        return splited_df_dict

    def _flatten(self, lst: list) -> list:
        return [x for xs in lst for x in xs]   




# TEST #
# todo remove on prod   
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dm = PneumoniaDataManager('../data/chest_xray')   
    train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ])

    print(dm.splited_df_dict['val'])
    val_dataset = PneumoniaDataset(dm.splited_df_dict['val'], train_transforms)
    image, label = val_dataset[0]
    plt.imshow( image.permute(1, 2, 0)  )
    plt.savefig("test.png")
