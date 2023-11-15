import argparse

import torch
from torch.backends import cudnn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Configuration of testing process")
parser.add_argument('-m', '--model', type=str,default='./model/RestNet18.pt')
parser.add_argument('-depth', default=18, type=int)
parser.add_argument('-d', '--data', type=str, default='')
parser.add_argument('-att_type', default='se', choices=['cbam', 'se'], type=str)
args = parser.parse_args()

transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
test_path = args.data + '/' + 'test'
dataset = Plain_Dataset( img_dir=test_path, datatype='test',transform=transformation)
test_loader = DataLoader(dataset,batch_size=64,num_workers=0)

# 加载模型
net = ResidualNet('CIFAR10', args.depth, 7, args.att_type)
net.load_state_dict(torch.load(args.model))
net.to(device)


# 混淆矩阵定义
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_maxtrix(maxtrix, per_kinds):
    # 分类标签
    lables = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise','Contempt']

    Maxt = np.empty(shape=[0, 8])

    m = 0
    for i in range(8):
        print('row sum:', per_kinds[m])
        f = (maxtrix[m, :] * 100) / per_kinds[m]
        Maxt = np.vstack((Maxt, f))
        m = m + 1

    thresh = Maxt.max() / 1

    plt.imshow(Maxt, cmap=plt.cm.Blues)

    for x in range(7):
        for y in range(7):
            info = float(format('%.1f' % F[y, x]))
            print('info:', info)
            plt.text(x, y, info, verticalalignment='center', horizontalalignment='center')
    plt.tight_layout()
    plt.yticks(range(7), lables)  # y轴标签
    plt.xticks(range(7), lables, rotation=45)  # x轴标签
    plt.savefig('./test.png', bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
    plt.show()


if __name__ == '__main__':
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)

            conf_maxtri = confusion_matrix(classs, labels, conf_maxtri)
            conf_maxtri = conf_maxtri.cpu()

            wrong = torch.where(classs != labels, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
            acc = 1 - (torch.sum(wrong) / 64)  # 64为batch size
            total.append(acc.item())

print('测试集的准确率为: %f %%' % (100 * np.mean(total)))

# 绘制混淆矩阵
conf_maxtri = np.array(conf_maxtri.cpu())
corrects = conf_maxtri.diagonal(offset=0)
per_kinds = conf_maxtri.sum(axis=1)
plot_maxtrix(conf_maxtri, per_kinds)
