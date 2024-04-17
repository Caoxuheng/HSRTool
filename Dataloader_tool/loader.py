
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import torch
import cv2
def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data
class PaviaUDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=25, patch_w=25, h_stride=10, w_stride=10, ratio=4, type='train'):
        super(PaviaUDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.hsi_channel=103
        self.rows = 110
        self.cols = 85
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels

        self.hrhsi, self.lrhsi = self.getData()
        if self.type=='train':
           self.hsi_data,  self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio)
        if self.type=='eval':
            self.hsi_data,self.label = self.generateEval(patch_h=self.patch_h, patch_w=self.patch_w, ratio=ratio)
        if self.type == 'test':
            self.hsi_data, self.label = self.generateTest(patch_h=self.patch_h, patch_w=self.patch_w, ratio=ratio)

    def getData(self):
        hrhsi = sio.loadmat(self.mat_save_path)['paviaU']
        hrhsi =hrhsi/ hrhsi.max()
        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi,ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
        return hrhsi, lrhsi

    def generateTrain(self, patch_h, patch_w, ratio):
        num = len(list(range(0, self.rows - patch_h , self.h_stride) ))*len(list(range(0, self.cols - patch_w , self.w_stride)))
        print('Train_num:',num)
        label_patch = np.zeros((num, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((num, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        # hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = self.hrhsi[:self.rows*ratio, :, :]
        lrhsi = self.lrhsi[:self.rows, :, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)

        for x in range(0, self.rows - patch_h , self.h_stride):
            for y in range(0, self.cols - patch_w , self.w_stride):

                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]

                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0


        hrhsi = self.hrhsi[self.rows * ratio:(self.rows+patch_w) * ratio, :patch_w*ratio, :]
        lrhsi = self.lrhsi[self.rows:self.rows+patch_w, :patch_w, :]
        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        label_patch[count] = hrhsi

        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, label_patch
    def generateTest(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0
        hrhsi = self.hrhsi[self.rows * ratio:(self.rows+patch_w) * ratio, patch_w*ratio:patch_w*ratio*2, :]
        lrhsi = self.lrhsi[self.rows:self.rows+patch_w, patch_w:2*patch_w, :]
        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        label_patch[count] = hrhsi

        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, label_patch

    def __getitem__(self, index):
        index =index
        hrhsi = np.transpose(self.label[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]

class PaviaDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=25, patch_w=25, h_stride=10, w_stride=10, ratio=4, type='train'):
        super(PaviaDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.hsi_channel=102
        self.rows = 200
        self.cols = 150
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels

        self.hrhsi, self.lrhsi = self.getData()
        if self.type=='train':
           self.hsi_data,  self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio)
        if self.type=='eval':
            self.hsi_data,self.label = self.generateEval(patch_h=self.patch_h, patch_w=self.patch_w, ratio=ratio)
        if self.type == 'test':
            self.hsi_data, self.label = self.generateTest(patch_h=self.patch_h, patch_w=self.patch_w, ratio=ratio)

    def getData(self):
        hrhsi = sio.loadmat(self.mat_save_path)['pavia']
        hrhsi =hrhsi/ hrhsi.max()

        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi,ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
        return hrhsi, lrhsi

    def generateTrain(self, patch_h, patch_w, ratio):
        num = len(list(range(0, self.rows - patch_h , self.h_stride) ))*len(list(range(0, self.cols - patch_w , self.w_stride)))
        print('Train_num:',num)
        label_patch = np.zeros((num, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((num, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        # hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = self.hrhsi[:self.rows*ratio, :, :]
        lrhsi = self.lrhsi[:self.rows, :, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)

        for x in range(0, self.rows - patch_h , self.h_stride):
            for y in range(0, self.cols - patch_w , self.w_stride):

                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]

                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0


        hrhsi = self.hrhsi[self.rows * ratio:(self.rows+patch_w) * ratio, :patch_w*ratio, :]
        lrhsi = self.lrhsi[self.rows:self.rows+patch_w, :patch_w, :]
        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        label_patch[count] = hrhsi

        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, label_patch
    def generateTest(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0
        hrhsi = self.hrhsi[self.rows * ratio:(self.rows+patch_w) * ratio, patch_w*ratio:patch_w*ratio*2, :]
        lrhsi = self.lrhsi[self.rows:self.rows+patch_w, patch_w:2*patch_w, :]
        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        label_patch[count] = hrhsi

        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, label_patch

    def __getitem__(self, index):
        index =index
        hrhsi = np.transpose(self.label[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]