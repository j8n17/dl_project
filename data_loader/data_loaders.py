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
import random

class CustomSampler(Sampler):
    def __init__(self, root_path, data_txt, shuffle):
        with open(os.path.join(root_path, data_txt), 'r') as file:
            self.datas = file.readlines()
        # 빈 딕셔너리 생성
        self.parent_dirs = {}

        # 중복되는 부모 디렉토리인 인덱스들을 리스트로 묶어서 딕셔너리에 저장
        for i, data in enumerate(self.datas):
            data = data.split()
            parent_dir = os.path.split(data[0])[0]
            if parent_dir not in self.parent_dirs:
                self.parent_dirs[parent_dir] = []
            self.parent_dirs[parent_dir].append(i)
            
        self.num_parent_dir = len(self.parent_dirs)
        self.shuffle = shuffle

    def __iter__(self):
        # 같은 부모 디렉토리인 인덱스들 중 하나만 샘플링.
        indices = []
        for same_parent_indices in self.parent_dirs.values():
            indices.append(random.choice(same_parent_indices))
        
        indices = np.array(indices)

        if self.shuffle:
            np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.num_parent_dir

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
        print(data) # 데이터 샘플링 확인하기 위한 코드, 삭제 예정.
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

        return frames, label

class CustomDataLoader(BaseDataLoader):
    def __init__(self, root_path, data_txt, batch_size, shuffle=True, num_workers=1, mode='train'):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), # Normalize 값 탐색 후 변경
            transforms.Resize((224, 224)) # 나중에 수정 예정
        ])
        self.root_path = root_path
        self.data_txt = data_txt
        self.dataset = CustomDataset(self.root_path, self.data_txt, transforms=trsfm)
        if mode == 'train':
            self.sampler = CustomSampler(self.root_path, self.data_txt, shuffle=shuffle)
            shuffle = False # CustomSampler를 사용할 때는 DataLoader의 shuffle은 False여야 함.
        else:
            self.sampler = None
        super().__init__(self.dataset, batch_size, shuffle, num_workers, self.sampler)
