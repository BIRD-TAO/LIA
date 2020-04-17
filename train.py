import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from model import DCNet_Classifier,LIANet
from utils import prepare_data,regularization_term
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=128)
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--outf', default='outputs/', help='folder to output images and model checkpoints')
parser.add_argument('--gamma', default=3,help='weight of regularization term. default=3')
parser.add_argument('--alpha', default=0.001,help='weight of feature extraction term. default=0.001')

opt = parser.parse_args()
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#图像读入与预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale(opt.imageSize), #128
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms) #data/

dataloader = torch.utils.data.DataLoader( 
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)


LIA = LIANet(opt.ngf, opt.nz).to(device)
netD = DCNet_Classifier(opt.ndf).to(device)

# 不训练 EN_f
for para in LIA.EN_f.parameters():
    para.requires_grad=False


Loss_CrossEn = nn.BCELoss()
optimizerG = torch.optim.Adam(filter(lambda para: para.requires_grad, LIA.parameters()), lr=0.002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
'''
use GAN training stategy to get optimal g and D
'''

for epoch in range(1, opt.epoch + 1):
    for i, (imgs,_) in enumerate(dataloader):
        # 固定生成器g，训练鉴别器D
        optimizerD.zero_grad()
        ## 让D尽可能的把真图片判别为1
        imgs = imgs.to(device)
        imgs = prepare_data(imgs)
        output1 = netD(imgs)
        label.data.fill_(real_label)
        label=label.to(device)
        errD_real = Loss_CrossEn(output1, label)
        #errD_real.backward()

        ## 让D尽可能把假图片判别为0
        label.data.fill_(fake_label)
        noise = torch.randn(opt.batchSize, opt.nz)
        noise = noise.to(device)
        fake = LIA.g(noise)  # 生成假图
        output2 = netD(fake.detach()) #避免梯度传到G，因为G不用更新
        errD_fake = Loss_CrossEn(output2, label)
        #errD_fake.backward()
        
        ## 计算regularization项
        imgs.requires_grad = True
        output3 = netD(imgs)
        regularization = regularization_term(output3,imgs) ##用l2范数
        imgs.requires_grad = False

        ##计算L = errD
        errD = errD_fake + errD_real + opt.gamma/2*regularization
        errD.backward()
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label = label.to(device)
        output4 = netD(fake)
        errG = Loss_CrossEn(output4, label)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

    noise = torch.randn(opt.batchSize, opt.nz)
    noise = noise.to(device)
    fake = LIA.g(noise)
    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)


'''
encoder training to get optimal f
'''
# frozen g parameters
for para in LIA.EN_f.parameters():
    para.requires_grad=True
for para in LIA.coupling.parameters():
    para.requires_grad=False
for para in LIA.scaling.parameters():
    para.requires_grad=False
for para in LIA.DC_g.parameters():
    para.requires_grad=False
for para in netD.parameters():
    para.requires_grad=False
    
optimizerG = torch.optim.Adam(filter(lambda para: para.requires_grad, LIA.parameters()), lr=0.002, betas=(0.5, 0.999))

model_vgg = torchvision.models.vgg11(pretrained=True)
for para in model_vgg.parameters():
    para.requires_grad = False

for epoch in range(opt.epoch//2 + 1):
    for i, (imgs,_) in enumerate(dataloader):
        optimizerG.zero_grad()
        imgs = imgs.to(device)
        imgs = prepare_data(imgs)
        
        ## 真实图片为1的概率
        output1 = netD(imgs)
        label.data.fill_(real_label)
        label=label.to(device)
        err_real = Loss_CrossEn(output1, label)
        
        ## 虚假图片为0的概率
        label.data.fill_(fake_label)
        fake = LIA(imgs)  # 生成假图
        output2 = netD(fake.detach()) #避免梯度传到G，因为G不用更新
        err_fake = Loss_CrossEn(output2, label)
        
        ## 计算regularization项
        imgs.requires_grad = True
        output3 = netD(imgs)
        regularization = regularization_term(output3,imgs) ##用l2范数
        imgs.requires_grad = False
        
        ##计算feture exactor项
        output3 = model_vgg(imgs)
        fake = LIA(imgs)
        output4 = model_vgg(fake)
        Loss_FE = (output1 - output2).pow(2).sum().sqrt() ## 用l2范数
        
        err_overall = err_real + err_fake + opt.gamma/2*regularization + opt.alpha*Loss_FE
        err_overall.backward()
        optimizerG.step()
        print('[%d/%d][%d/%d] Loss_F: %.3f Loss_FE %.3f'
              % (epoch, opt.epoch, i, len(dataloader), err_fake.item(), Loss_FE.item()))

torch.save(LIA.state_dict(), '%s/LIA.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))

