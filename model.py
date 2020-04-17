import torch.nn as nn
import torch

class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim = None, hidden_num = 0, mask_config = 1):
        """Initialize a coupling layer.
        Args:
            in_out_dim: 输入与输出的维度（NF中输入与输出相同）
            mid_dim: 隐藏层的维度
            hidden_num: 隐藏层数目
            mask_config: 1 对index为奇数的进行转换, 0 对index为偶数对进行转换。
        """
        super(Coupling, self).__init__()

        mid_dim = in_out_dim if mid_dim is None else mid_dim
        if hidden_num == 0:
            assert (mid_dim == in_out_dim)
            
        self.hidden_num = hidden_num
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),
            nn.ReLU())

        if hidden_num >= 1:
            self.mid_block = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                    ) for _ in range(hidden_num - 1)])

        self.out_block = nn.Linear(mid_dim, in_out_dim//2)

    def forward(self, x, reverse=False):
        """Forward .
        Args:
            x: 输入张量.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            转换后的张量。
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        if self.hidden_num >1 :
            for i in range(len(self.mid_block)):
                off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))

class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x

class DCNet_Generator(nn.Module):
    def __init__(self, ngf, nz):
        super(DCNet_Generator, self).__init__()
        # layer1输入的是一个128x1x1的随机噪声, 输出尺寸(ngf*16)x4x4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(inplace=True)
        )
        # layer2输出尺寸(ngf*8)x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        # layer3输出尺寸(ngf*4)x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # layer4输出尺寸(ngf*2)x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        # layer5输出尺寸 ngf x 64 x 64
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU()
        )
        # layer6输出尺寸 3 x 128 x 128
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # 定义NetG的前向传播

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out

# 定义鉴别器网络C
class DCNet_Classifier(nn.Module):
    def __init__(self, ndf):
        super(DCNet_Classifier, self).__init__()
        # layer1 输入 3 x 128 x 128, 输出 (ndf) x 64 x 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2 输出 (ndf*2) x 32 x 32
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3 输出 (ndf*4) x 16 x 16
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4 输出 (ndf*8) x 8 x 8
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 输出 (ndf*16) x 4 x 4
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer6 输出一个数(概率)
        self.layer6 = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # 定义NetD的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out

#定义f网络

class fNet(nn.Module):
    def __init__(self, ngf, nz):
        super(fNet, self).__init__()
        # 输入ngf x 64 x 64,输出 (ngf*2) x 32 x 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
        )
        # 输入ngf x 64 x 64,输出 (ngf*2) x 32 x 32
        self.layer2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
        )
        # layer3输出尺寸(ngf*4)x16x16
        self.layer3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU()
        )
        # 输出(ngf*8)x8x8
        self.layer4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
        )
        # 输出(ngf*16)x4x4
        self.layer5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU()
        )
        # 输出128 x 1 x 1
        self.layer6 = nn.Sequential(
            nn.Conv2d(ngf * 16, nz, 4, 1, 0, bias=False)
        )

        # 定义Netf的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


'''定义LIANet:
结构为:
x :[B,C*H*W] -> EN_f -> y -> coupling1 -> coupling2 -> ... -> coupling4 ->  Scaling -> z -> inverse_Scaling
inverse_coupling4 -> inverse_coupling3 -> .. -> inverse_coupling1 -> y'  -> DCGAN_Generator
 '''

class LIANet(nn.Module):
    def __init__(self, ngf, nz, coupling_k = 4, mid_dim = None, hidden_num = 0, mask_config = 1):
        """Forward.
               Args:
                   nz: z的维度，这里设置为128，也是coupling中in_out_input的维度.
                    coupling_k: coupling层的数目
                    coupling中的参数:
                        mid_dim: 隐藏层的维度
                        hidden_num: 隐藏层数目
                        mask_config: 1 对index为奇数的进行转换, 0 对index为偶数对进行转换。
                    DCGAN中的参数：
                        ngf: 中间卷积的channel数目单位.
               Returns:
                    3x128x128的图片。
         """
        super(LIANet, self).__init__()
        self.nz = nz
        self.EN_f = fNet(ngf,nz)
        
        self.coupling = nn.ModuleList([
            Coupling(in_out_dim=nz,
                     mid_dim=mid_dim,
                     hidden_num=hidden_num,
                     mask_config=(mask_config + i) % 2) \
            for i in range(coupling_k)])
        self.scaling = Scaling(nz)
        self.DC_g = DCNet_Generator(ngf,nz)

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).
        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)
        
        [B, C] = list(x.size())
        x = x.reshape(B,C,1,1)
        x = self.DC_g(x)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        x = self.EN_f(x)
        assert len(list(x.size())) == 4
        [B, C, H, W] = list(x.size())
        x = x.reshape((B, C * H * W))
        
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
        return self.scaling(x)

    def forward(self, x):
        x = self.f(x)
        x = self.g(x)
        return x
    
    def sampling(self,z):
        '''
        Args:
            z: torch.randn((sample_number,128))
        Returns:
            imgs: size = (sample_number,3,128,128)
        '''
        assert (z.size() == 4)
        return self.g(z)



