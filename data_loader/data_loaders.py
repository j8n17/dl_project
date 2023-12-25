from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import torch
from torch.utils.data import Sampler
import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt

class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # 여기에 샘플링 로직을 정의합니다.
        # 예시로 데이터셋의 인덱스를 무작위로 섞은 리스트를 반환합니다.
        indices = np.random.permutation(len(self.data_source))
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

class CustomDataset(Dataset):
    def __init__(self, root_path, data_txt, transforms):
        # 데이터셋 초기화
        # 예: 파일 경로 목록 로드, 변환 초기화 등
        self.root_path = root_path
        self.data_txt = data_txt
        self.transforms = transforms

        with open(os.path.join(self.root_path, self.data_txt), 'r') as file:
            self.data = file.readlines()

    def __len__(self):
        # 데이터셋의 전체 크기 반환
        return len(self.data)

    def __getitem__(self, idx):
        # idx에 해당하는 데이터 로드 및 처리
        # 예: 이미지를 로드하고 변환 적용
        data = self.data[idx].split()
        frame_folder = os.path.join(self.root_path, data[0])
        clips = data[1]
        label = torch.tensor(int(data[2]))

        frames_path = sorted(glob(os.path.join(frame_folder, '*.jpg'))) # 오름차순으로 정렬
        frame_ls = []
        for frame_path in frames_path:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transforms(frame)
            frame_ls.append(frame)
        frames = torch.stack(frame_ls)

        return frame[0].reshape(1, 28, 28), label

class CustomDataLoader(BaseDataLoader):
    def __init__(self, root_path, data_txt, batch_size, shuffle=True, num_workers=1, mode='train'):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), # Normalize 값 탐색 후 변경
            transforms.Resize((28, 28)) # 나중에 수정 예정
        ])
        self.root_path = root_path
        self.data_txt = data_txt
        self.dataset = CustomDataset(self.root_path, self.data_txt, transforms=trsfm)
        if mode == 'train':
            self.sampler = None #CustomSampler()
        else:
            self.sampler = None
        super().__init__(self.dataset, batch_size, shuffle, num_workers, self.sampler)
