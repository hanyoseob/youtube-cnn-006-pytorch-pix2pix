
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms

def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt = args.wgt
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))

    if not os.path.exists(result_dir_val):
        os.makedirs(os.path.join(result_dir_val, 'png'))

    ## 네트워크 학습하기
    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(286, 286, nch)),
                                              RandomCrop((ny, nx)),
                                              Normalization(mean=0.5, std=0.5)])

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'),
                                transform=transform_train,
                                task=task, opts=opts)
        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)


        transform_val = transforms.Compose([Resize(shape=(286, 286, nch)),
                                            RandomCrop((ny, nx)),
                                            Normalization(mean=0.5, std=0.5)])

        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'),
                              transform=transform_val,
                              task=task, opts=opts)
        loader_val = DataLoader(dataset_val, batch_size=batch_size,
                                shuffle=True, num_workers=8)

        num_data_val = len(dataset_val)
        num_batch_val = np.ceil(num_data_val / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "pix2pix":
        netG = Pix2Pix(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=2 * nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    # fn_loss = nn.MSELoss().to(device)

    fn_l1 = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                                        netG=netG, netD=netD,
                                                        optimG=optimG, optimD=optimD)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netG.train()
            netD.train()

            loss_G_l1_train = []
            loss_G_gan_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            for batch, data in enumerate(loader_train, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                # input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

                output = netG(input)

                # backward netD
                set_requires_grad(netD, True)
                optimD.zero_grad()

                real = torch.cat([input, label], dim=1)
                fake = torch.cat([input, output], dim=1)

                pred_real = netD(real)
                pred_fake = netD(fake.detach())

                loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad(netD, False)
                optimG.zero_grad()

                fake = torch.cat([input, output], dim=1)
                pred_fake = netD(fake)

                loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
                loss_G_l1 = fn_l1(output, label)
                loss_G = loss_G_gan + wgt * loss_G_l1

                loss_G.backward()
                optimG.step()

                # 손실함수 계산
                loss_G_l1_train += [loss_G_l1.item()]
                loss_G_gan_train += [loss_G_gan.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN L1 %.4f | GEN GAN %.4f | "
                      "DISC REAL: %.4f | DISC FAKE: %.4f" %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_G_l1_train), np.mean(loss_G_gan_train),
                       np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

                if batch % 20 == 0:
                  # Tensorboard 저장하기
                  input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                  label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                  output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                  input = np.clip(input, a_min=0, a_max=1)
                  label = np.clip(label, a_min=0, a_max=1)
                  output = np.clip(output, a_min=0, a_max=1)

                  id = num_batch_train * (epoch - 1) + batch

                  plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0], cmap=cmap)
                  plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0], cmap=cmap)
                  plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

                  writer_train.add_image('input', input, id, dataformats='NHWC')
                  writer_train.add_image('label', label, id, dataformats='NHWC')
                  writer_train.add_image('output', output, id, dataformats='NHWC')

            writer_train.add_scalar('loss_G_l1', np.mean(loss_G_l1_train), epoch)
            writer_train.add_scalar('loss_G_gan', np.mean(loss_G_gan_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

            with torch.no_grad():
                netG.eval()
                netD.eval()

                loss_G_l1_val = []
                loss_G_gan_val = []
                loss_D_real_val = []
                loss_D_fake_val = []

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)
                    # input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

                    output = netG(input)

                    # backward netD
                    # set_requires_grad(netD, True)
                    # optimD.zero_grad()

                    real = torch.cat([input, label], dim=1)
                    fake = torch.cat([input, output], dim=1)

                    pred_real = netD(real)
                    pred_fake = netD(fake.detach())

                    loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
                    loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    # loss_D.backward()
                    # optimD.step()

                    # backward netG
                    # set_requires_grad(netD, False)
                    # optimG.zero_grad()

                    fake = torch.cat([input, output], dim=1)
                    pred_fake = netD(fake)

                    loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
                    loss_G_l1 = fn_l1(output, label)
                    loss_G = loss_G_gan + wgt * loss_G_l1

                    # loss_G.backward()
                    # optimG.step()

                    # 손실함수 계산
                    loss_G_l1_val += [loss_G_l1.item()]
                    loss_G_gan_val += [loss_G_gan.item()]
                    loss_D_real_val += [loss_D_real.item()]
                    loss_D_fake_val += [loss_D_fake.item()]

                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | "
                          "GEN L1 %.4f | GEN GAN %.4f | "
                          "DISC REAL: %.4f | DISC FAKE: %.4f" %
                          (epoch, num_epoch, batch, num_batch_val,
                           np.mean(loss_G_l1_val), np.mean(loss_G_gan_val),
                           np.mean(loss_D_real_val), np.mean(loss_D_fake_val)))

                    if batch % 10 == 0:
                        # Tensorboard 저장하기
                        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                        label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                        output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                        input = np.clip(input, a_min=0, a_max=1)
                        label = np.clip(label, a_min=0, a_max=1)
                        output = np.clip(output, a_min=0, a_max=1)

                        id = num_batch_train * (epoch - 1) + batch

                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_input.png' % id), input[0], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_label.png' % id), label[0], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

                        writer_val.add_image('input', input, id, dataformats='NHWC')
                        writer_val.add_image('label', label, id, dataformats='NHWC')
                        writer_val.add_image('output', output, id, dataformats='NHWC')

                writer_val.add_scalar('loss_G_l1', np.mean(loss_G_l1_val), epoch)
                writer_val.add_scalar('loss_G_gan', np.mean(loss_G_gan_val), epoch)
                writer_val.add_scalar('loss_D_real', np.mean(loss_D_real_val), epoch)
                writer_val.add_scalar('loss_D_fake', np.mean(loss_D_fake_val), epoch)

            if epoch % 50 == 0 or epoch == num_epoch:
                save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

        writer_train.close()
        writer_val.close()

def test(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt = args.wgt
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))
        os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기
    if mode == "test":
        transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "pix2pix":
        netG = Pix2Pix(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=2 * nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)


    ## 손실함수 정의하기
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    # fn_loss = nn.MSELoss().to(device)

    fn_l1 = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == "test":
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

        with torch.no_grad():
            netG.eval()

            loss_G_l1_test = []

            for batch, data in enumerate(loader_test, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                # input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

                output = netG(input)

                loss_G_l1 = fn_l1(output, label)

                # 손실함수 계산
                loss_G_l1_test += [loss_G_l1.item()]

                print("TEST: BATCH %04d / %04d | GEN L1 %.4f" %
                      (batch, num_batch_test, np.mean(loss_G_l1_test)))

                # Tensorboard 저장하기
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                for j in range(label.shape[0]):

                    id = batch_size * (batch - 1) + j

                    input_ = input[j]
                    label_ = label[j]
                    output_ = output[j]

                    np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_)
                    np.save(os.path.join(result_dir_test, 'numpy', '%04d_label.npy' % id), label_)
                    np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                    input_ = np.clip(input_, a_min=0, a_max=1)
                    label_ = np.clip(label_, a_min=0, a_max=1)
                    output_ = np.clip(output_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

            print('AVERAGE TEST: GEN L1 %.4f' % np.mean(loss_G_l1_test))