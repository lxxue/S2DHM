import numpy as np
from PIL import Image
from pathlib import Path
from glob import glob
import scipy.io
import h5py  # for loading v7.3 .mat

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.transforms import transforms
import torch

"""
Train: For each image pair creates randomly positive and negative matches
    cross-season-correspondence dataset, Robotcar
    todo: check if all image pairs are in query list  

Initialize Robotcar class attributes.
        Args:
            root: The root to the dataset folder.
            image_folder: The folder containing the images.
            pair_info_folder: The folder containing .mat file 
                        that stores image pairs and their positive correspondences.
            name: The dataset name.
            image_pairs_name: The dict storing the path(name) to image pairs.
            corres_pos_all: The dict storing all the positive correspondences given by .mat files. 
"""


class RobotcarCorrDataset(Dataset):
    def __init__(self, root: str,
                 image_folder: str,
                 pair_info_folder: str,
                 name: str = None,
                 queries_folder: str = None,
                 robotcar_weather_all: bool = True,
                 robotcar_weather: str = None,
                 transform=None,
                 img_scale: int = None,
                 num_matches: int = None
                 ):
        self._data = {
            'name': 'robotcar',
            'root': root,
            'image_folder': image_folder,
            'pair_info_folder': pair_info_folder,
            'pair_file_names': None,
            'queries_folder': queries_folder,
            'image_pairs_name': None,
            'corres_pos_all': None,
            'scale': img_scale,
            'num_matches': num_matches
        }
        self.load_pair_file_names(robotcar_weather, robotcar_weather_all)
        self.load_image_pairs()
        self.transform = transform
        self.default_transform = self.default_transform()

    def load_pair_file_names(self, robotcar_weather, robotcar_weather_all):
        # load image pairs for one slice
        # /public_data/robotcar/corrrespondence
        pair_file_roots = Path(self._data['root'], self._data['name'], self._data['pair_info_folder'])
        # /public_data/robotcar/corrrespondence/*.mat
        if not robotcar_weather_all:
            suffix = 'correspondence_run1_overcast-reference_run2_{}*.mat'.format(robotcar_weather)
        else:
            suffix = '*.mat'
        pair_files = glob(str(Path(pair_file_roots, suffix)))
        if not len(pair_files):
            raise Exception('No correspondence file found at {}'.format(pair_file_roots))
        if not robotcar_weather_all:
            print('>> Found {} image pairs for weather {}'.format(len(pair_files), robotcar_weather))
        else:
            print('>> Found {} image pairs for Robotcar dataset'.format(len(pair_files)))
        
        self._data['pair_file_names'] = pair_files

# 

    def load_image_pairs(self):
        N = len(self._data['pair_file_names'])  # number of image pairs
        image_pairs = {'a': [], 'b': []}
        corres_all_pos = {'a': [], 'b': []}
        num_pairs = len(self._data['pair_file_names'])
        poses = np.zeros((num_pairs, 4, 4))
        for i, f in enumerate(self._data['pair_file_names']):
            # pair_info = scipy.io.loadmat(f)
            # pair_info = h5py.File(f, 'r+')
            pair_info = scipy.io.loadmat(f)
            org_name_a = pair_info['im_i_path'][0]
            org_name_b = pair_info['im_j_path'][0]
            # org_name_a = u''.join(chr(c) for c in pair_info['im_i_path'])
            # org_name_b = u''.join(chr(c) for c in pair_info['im_j_path'])
            # query_root = Path(self._data['root'], self._data['name'], self._data['image_folder'],
            #                   (f.split('/')[-1]).split('_')[1], self._data['queries_folder'])
            # query_root = Path(self._data['root'], self._data['name'], self._data['image_folder'])
            query_root = Path(self._data['image_folder'])
            image_pairs['a'].append(Path(query_root, org_name_a))
            image_pairs['b'].append(Path(query_root, org_name_b))
            corres_all_pos['a'].append(pair_info['pt_i'].transpose())
            corres_all_pos['b'].append(pair_info['pt_j'].transpose())  # N x 2
            poses[i] = pair_info['pose']
        self._data['image_pairs_name'] = image_pairs
        self._data['corres_pos_all'] = corres_all_pos
        self._data['poses'] = poses

    # To do: double check image W, H !!!!!!
    '''?
    mean and std for normalization? 
    '''

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((1024 // self._data['scale'], 1024 // self._data['scale'])),
            # check image dim, resize (H, W)?  just for fast debugging
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.03001604,0.08044077,0.13968322], std=[1.0841591,1.0996625,1.1056131]), # all image in robotcar
        ])

    '''
    '''

    def __getitem__(self, idx):
        img_a = self._data['image_pairs_name']['a'][idx]
        img_b = self._data['image_pairs_name']['b'][idx]
        a = self._data['corres_pos_all']['a'][idx].squeeze()
        b = self._data['corres_pos_all']['b'][idx].squeeze()
        if self.transform:
            img_a = self.default_transform(Image.open(img_a))
            img_b = self.default_transform(Image.open(img_b))
        # pos_a, pos_b = random_select_positive_matches(a, b, num_of_pairs=self._data['num_matches'])
        # modified: return all pos matches
        corres_ab_pos = {'a': a, 'b': b}
        return (img_a, img_b), (corres_ab_pos)

    def __len__(self):
        assert len(self._data['image_pairs_name']['a']) == len(self._data['image_pairs_name']['b'])
        return len(self._data['image_pairs_name']['a'])

if __name__ == "__main__":
    dset = RobotcarCorrDataset(
        root="/local/home/lixxue/gnnet/robotcar_data/",
        image_folder="images",
        pair_info_folder="correspondence",
        queries_folder="query_all",
        img_scale=1,
    )
    print(dset._data['image_pairs_name']['a'][0])
    print(dset._data['image_pairs_name']['b'][0])
    print(dset._data['corres_pos_all']['a'][0].shape)
    print(dset._data['corres_pos_all']['b'][0].shape)