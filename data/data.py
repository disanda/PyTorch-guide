import torch
import os
from PIL import Image
from torchvision import datasets, transforms

class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.path = './pose_set_1'#指定自己的路径
        self.image_filenames = [x for x in os.listdir(self.path)]
    def __getitem__(self, index):
        a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('L')
        a = a.resize((64, 64), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        return a
    def __len__(self):
        return len(self.image_filenames)

pose = DatasetFromFolder()

train_loader = torch.utils.data.DataLoader(
     dataset=pose,
     batch_size=25,#一个batch25张图片,即一次epoch
     shuffle=False,
     #num_workers=0,若是win需要这一行
     pin_memory=True,#用Nvidia GPU时生效
     drop_last=True
 )

for i, x in enumerate(train_loader):
     print(i)
     print(x.shape)#[n,c,w,h]
     #torchvision.utils.save_image(x, './pose-img/%d.jpg'%(i), nrow=5)
