import shutil
import os

from torch.utils.data.dataloader import DataLoader
from torcheeg import model_selection
from torcheeg.model_selection import KFoldGroupbyTrial, KFoldPerSubject
from torcheeg.model_selection.subcategory import Subcategory

def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

class TempData:
    def __init__(self, data, gender):
        self.data = data
        self.test_id = list(set(data.info['trial_id']))
        self.gender = gender
    def __getitem__(self, item):
        return (self.data[item], self.test_id, self.gender)

class DataSetModule:
    def __init__(self, emotion, PREFIX, data_class):
        self.PREFIX = PREFIX
        self.data_name = data_class.name
        self.emotion = emotion

        self.dataset = data_class.data
        print("Read data done")
        reset_folder(self.PREFIX + '/split')
        print("Finish preparing data")
        self.train = []
        self.test = []
        self.train_test = list(self.setup())
        for train, test in self.train_test:
            self.train.append(train)
            self.test.append(test)
        self.male_female = []
        self.male = []
        self.female = []
        for idx, test in enumerate(self.test):
            self.male_female.append(list(self.gender_split(idx, test)))
        for idx, all_gender in enumerate(self.male_female):
            self.male.append(list(self.per_trial_split(idx, all_gender[0], 'male')))
            self.female.append(list(self.per_trial_split(idx, all_gender[1], 'female')))

    def setup(self):
        # train_dataset, test_dataset = model_selection.train_test_split_cross_trial(dataset=self.dataset,
        #                                                                            split_path=self.PREFIX + "/split/all_gender",
        #                                                                            shuffle=True)
        train_test_split = model_selection.KFoldCrossTrial(n_splits=10, split_path=self.PREFIX + "/split/train_test_split")
        return train_test_split.split(self.dataset)

    def gender_split(self, idx, test):
        gender_test_split = Subcategory('Gender', split_path=self.PREFIX + '/split/gender' + str(idx))
        return gender_test_split.split(test)

    def per_trial_split(self, idx, data, gender):
        trial_split = Subcategory('trial_id', split_path = self.PREFIX + '/split/trial_' + gender + str(idx))
        return trial_split.split(data)

    def train_loader(self, idx, batch_size):
        return DataLoader(self.train[idx], batch_size = batch_size, num_workers = 16, pin_memory = True)

    def gender_test_loader(self, idx, batch_size):
        data_male_female = [0, 0]
        data_male_female[0] = DataLoader(self.male_female[idx][0], batch_size=batch_size, shuffle=False, num_workers=16,
                                         pin_memory=True)
        data_male_female[1] = DataLoader(self.male_female[idx][1], batch_size=batch_size, shuffle=False, num_workers=16,
                                         pin_memory=True)
        return data_male_female
    
    def male_loader(self, batch_size):
        return DataLoader(self)
        