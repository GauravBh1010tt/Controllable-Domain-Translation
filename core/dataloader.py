import h5py
import numpy as np
import torch
import random

from tqdm import tqdm
#from munch import Munch
from PIL import Image, ImageDraw
from torchvision import transforms
import torch.nn.functional as F

from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms

class HDF5DataLoader(object):
    def __init__(
        self, hdf5_file, batch_size, split, n_test=2000, dataset='cub', aligned = False, device='cuda', resize=256, latents=8, prob=0.9):
        super().__init__()
        self.h5 = h5py.File(hdf5_file, "r")
        
        self.dataset = dataset
        self.aligned = aligned
        self.split = split
        self.resize = resize
        self.latents = latents
        self.prob = prob
        self.batch_size = batch_size
        
            
        if split == "train":
            
            self.male_img = self.h5["tr_male_data"]
            self.att = self.h5['tr_att']
            self.female_img = self.h5['tr_female_data']
            self.ref_m_img = self.h5['tr_ref_male_data']
            self.ref_f_img = self.h5['tr_ref_female_data']

            self.n_samples_im = len(self.male_img)
            self.n_samples_hed = len(self.male_img)
            self.n_samples_att = len(self.male_img)

            self.indices = np.arange(self.n_samples_im)
            np.random.shuffle(self.indices)

            self.male_ind = self.indices
            self.att_ind = self.indices
            self.female_ind = self.indices
            self.ref_m_ind = self.indices
            self.ref_f_ind = self.indices
            self.n_samples = len(self.male_img)
        else:            
            self.male_img = self.h5["val_male_data"]
            self.att = self.h5['val_att']
            self.female_img = self.h5['val_female_data']
            self.ref_m_img = self.h5['val_ref_male_data']
            self.ref_f_img = self.h5['val_ref_female_data']

            self.n_samples_im = len(self.male_img)
            self.n_samples_hed = len(self.male_img)
            self.n_samples_att = len(self.male_img)

            self.indices = np.arange(self.n_samples_im)
            
            self.male_ind = self.indices
            self.att_ind = self.indices
            self.female_ind = self.indices
            self.ref_m_ind = self.indices
            self.ref_f_ind = self.indices
            self.n_samples = len(self.male_img)

        self.split = split
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __len__(self):
        #if self.split == 'test':
        return self.n_samples // self.batch_size
        #else:
        #    return min(self.n_samples_im, self.n_samples_hed) // self.batch_size
    
    def transform(self, im=None, att=[]):
        # convert data from b,0,1,c to b,c,0,1
        
        #im = np.transpose(im, (0,3,1,2))
        
        #print (im.type)
        
        #print (im.shape)
        
        if self.resize != 256:
            im = F.interpolate(im, self.resize)
        
        #print (im.shape)
        #bre
        if len(att)>0:
            # scale attribute vector between -1 to 1
            att = 2.*(att - np.min(att))/np.ptp(att)-1
            
            return im.astype('float32'), att.astype('float32')
        else:
            return im


    def create_iterator(self):

        for i in range(0, self.batch_size * self.__len__(), self.batch_size):
            
            idx = sorted(self.indices[i : i + self.batch_size])

            # Normalize data
            #try:
            if True:
                male_img = self.transform(torch.from_numpy(self.male_img[idx].astype('float32')))
                female_img = self.transform(torch.from_numpy(self.female_img[idx].astype('float32')))
                ref_m_img = self.transform(torch.from_numpy(self.ref_m_img[idx].astype('float32')))
                ref_f_img = self.transform(torch.from_numpy(self.ref_f_img[idx].astype('float32')))
                att = self.att[idx]
                
                z_trg = torch.randn(self.batch_size, self.latents)
                z_trg2 = torch.randn(self.batch_size, self.latents)
                
            #except:
            else:
                print (i, im_idx, att_idx)
                bre
                #print ()
            if self.device == 'cuda':
                male_img = male_img.cuda()
                female_img = female_img.cuda()
                ref_m_img = ref_m_img.cuda()
                ref_f_img = ref_f_img.cuda()
                att = torch.from_numpy(att).cuda()
                z_trg = z_trg.cuda()
                z_trg2 = z_trg2.cuda()
                
            yield {'im_m':male_img,'im_fe':female_img, 'im_m2':ref_m_img, 'im_fe2':ref_f_img, 
                   'att':att,'z_trg':z_trg, 'z_trg2':z_trg2}
            #yield Munch('x_src'=male_img,'x_ref'=female_img, 'x_ref2'=ref_img, 'att'=att, 'z_trg'=z_trg, 'z_trg2'=z_trg2)

    def create_iterator_n_iters(self, n):
        iterator = self.create_iterator()
        for i in range(n):
            try:
                yield next(iterator)
            except StopIteration:
                iterator = self.create_iterator()
                yield next(iterator)




def create_onehot(att, att_info):
    a = np.zeros(len(att_info.keys()))
    
    ind = att_info[att]
    a[ind] = 1
    return a


# define custom dataloader from torch
class MSCOCO_att(Dataset):
    def __init__(self, data_path=None, att_info=None, resize=256):
        self.data_path = data_path
        self.att_info = att_info

        # Data transforms
        self.transform = transforms.Compose(
            [transforms.Resize((resize, resize)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.resize = resize

    def __len__(self):
        return len(self.data_path)
  
    def __getitem__(self, idx):
        
        [cat_im1, cat_bbox1], [cat_im2, cat_bbox2], [dog_im1, dog_bbox1], [dog_im2, dog_bbox2], att = self.data_path[idx]
        
        im_cat1, im_dog1 = Image.open(cat_im1), Image.open(dog_im1)
        im_cat2, im_dog2 = Image.open(cat_im2), Image.open(dog_im2)
        
        im_cat1 = transforms.functional.crop(im_cat1, cat_bbox1[1],cat_bbox1[0],cat_bbox1[3],cat_bbox1[2])
        im_dog1 = transforms.functional.crop(im_dog1, dog_bbox1[1],dog_bbox1[0],dog_bbox1[3],dog_bbox1[2])
        
        im_cat2 = transforms.functional.crop(im_cat2, cat_bbox2[1],cat_bbox2[0],cat_bbox2[3],cat_bbox2[2])
        im_dog2 = transforms.functional.crop(im_dog2, dog_bbox2[1],dog_bbox2[0],dog_bbox2[3],dog_bbox2[2])
       
        cat_tensor1, dog_tensor1 = self.transform(im_cat1), self.transform(im_dog1)
        cat_tensor2, dog_tensor2 = self.transform(im_cat2), self.transform(im_dog2)
        
        att_label = torch.Tensor(create_onehot(att, self.att_info))

        return cat_tensor1.cuda(), cat_tensor2.cuda(), dog_tensor1.cuda(), dog_tensor2.cuda(), att_label.cuda()
    
    
class AFHQ(Dataset):
    def __init__(self, data_path=None, att_info=None, resize=256):
        self.data_path = data_path
        self.att_info = att_info

        # Data transforms
        self.transform = transforms.Compose(
            [transforms.Resize((resize, resize)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.resize = resize

    def __len__(self):
        return len(self.data_path)
  
    def __getitem__(self, idx):
        
        cat_im1, cat_im2, dog_im1, dog_im2 = self.data_path[idx]
        
        im_cat1, im_dog1 = Image.open(cat_im1), Image.open(dog_im1)
        im_cat2, im_dog2 = Image.open(cat_im2), Image.open(dog_im2)
        
        cat_tensor1, dog_tensor1 = self.transform(im_cat1), self.transform(im_dog1)
        cat_tensor2, dog_tensor2 = self.transform(im_cat2), self.transform(im_dog2)
        
        att_label = torch.zeros(self.att_info)

        return cat_tensor1.cuda(), cat_tensor2.cuda(), dog_tensor1.cuda(), dog_tensor2.cuda(), att_label.cuda()
                
'''
if __name__ == "__main__":
    loader = HDF5DataLoader(
        "data/cub/cub.hdf5", 16, 8192, "train"
    )

    while True:
        for i, x_t in tqdm(enumerate(loader.create_iterator())):
            if i == 0:
                print(x_t.shape)
'''
