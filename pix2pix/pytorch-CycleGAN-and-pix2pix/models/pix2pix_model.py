import os

import torch
from torch import nn
from .base_model import BaseModel
from models import create_model
from . import networks
from itertools import chain
import copy


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A',  'real_B', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        if self.opt.scaled_discriminator:
            self.model_names.append('small_D')
            self.pool = torch.nn.AvgPool2d(2)
        # define networks (both generator and discriminator)

        self.netG = networks.define_G(opt.input_nc + opt.noise, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if opt.inner > 0:
            self.model_names.append('small_G')
            self.netsmall_G = networks.define_G(opt.input_nc + opt.noise, opt.output_nc, opt.ngf*4, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            load_path = os.path.join(self.save_dir, "latest_net_small_G.pth")
            state_dict = torch.load(load_path, map_location=str(self.device))
            self.netsmall_G.module.load_state_dict(state_dict)
            self.small_G = nn.DataParallel(
                nn.Sequential(*[self.netsmall_G.module.down_model, self.netsmall_G.module.middle_model, self.netsmall_G.module.up_model]))

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if opt.FMLoss_weight > 0:
                FML = True
            else:
                FML = False
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, FML)

            self.netsmall_D = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,  opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, FML)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            if opt.FMLoss_weight > 0:
                self.criterionFM = networks.FMLoss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.inner > 1:
                params = chain(self.netG.parameters(), self.netsmall_G.parameters())
            else:
                params = self.netG.parameters()
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_smallD = torch.optim.Adam(self.netsmall_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_smallD)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward_net(self, net, real_A):
        size = real_A.size()[0]
        noise = torch.Tensor(torch.randn(size, self.opt.noise, 1, 1)).repeat(1, 1, self.opt.crop_size, self.opt.crop_size).to(self.device)
        real_A = torch.cat((real_A, noise), 1)
        if self.opt.inner > 0:
            return net(real_A, self.small_G)
        return net(real_A, None)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.forward_net(self.netG, self.real_A)  # G(A)

    def backward_net_D(self, netD, real_A, fake_B, real_B):
        # Fake; stop backprop to the generator by detaching big_fake_B
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        return loss_D_fake, loss_D_real, loss_D

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.opt.scaled_discriminator:
            self.backward_net_D(self.netsmall_D, self.pool(self.real_A), self.pool(self.fake_B), self.pool(self.real_B))
        self.loss_D_fake, self.loss_D_real, self.loss_D = self.backward_net_D(self.netD, self.real_A,
                                                                              self.fake_B, self.real_B)

    def backward_net_G(self, netD, real_A, fake_B, real_B):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB)

        if self.opt.FMLoss_weight > 0:
            self.criterionFM.set_fake_feature(netD.module)

        loss_G_GAN = self.criterionGAN(pred_fake, True)
        if self.opt.scaled_discriminator:
            pred_fake = self.netsmall_D(fake_AB)
            loss_G_GAN += self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_L1

        # third, get true targets for Feature matching loss
        if self.opt.FMLoss_weight > 0:
            real_AB = torch.cat((real_A, real_B), 1)
            _ = netD(real_AB)
            self.criterionFM.set_real_feature(netD.module)

        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        if self.opt.FMLoss_weight > 0:
            loss_FM = self.opt.FMLoss_weight*self.criterionFM()
            loss_G += loss_FM
        loss_G.backward()
        return loss_G_GAN, loss_G_L1, loss_G

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_G_GAN, self.loss_G_L1, self.loss_G = self.backward_net_G(self.netD, self.real_A, self.fake_B, self.real_B)


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.optimizer_smallD.zero_grad()
        self.backward_D()                # calculate gradients for D
        self.optimizer_smallD.step()
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
