import torch
import numpy as np
from torch.autograd import Variable # 获取变量

# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=4,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,4,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,4,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,4,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(18*64,100)
        self.mlp2 = torch.nn.Linear(100,1536)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x

if __name__ == '__main__':

    model = CNNnet()

    entity_embedding_final=np.load('/home/chenjianying/multimodalNER/visual_embedding/UMT/data/entity_embedding_final.npy', allow_pickle=True)
    entity_embedding_final = torch.from_numpy(entity_embedding_final)
    x = Variable(entity_embedding_final)

    out = model(x)

    print(out.shape)

