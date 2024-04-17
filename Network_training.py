from Networks.SSPSR import SSPSR
import scipy.io as sio
import os
from torch.utils.data import DataLoader
from torch import nn
import imgvision as iv
import numpy as np
from utils import PSNR_GPU, save_checkpoint
import argparse
from Networks.common import default_conv
import torch

# ========================Select fusion mode=====================
def Supervisedfusion(model, training_data_loader, validate_data_loader, model_folder, optimizer, lr,
                     start_epoch=0,end_epoch=2000, ckpt_step=50, RESUME=False):
    PLoss = nn.L1Loss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100,
                                                   gamma=0.5)
    print('Start training...')

    if RESUME:
        path_checkpoint = model_folder + "{}.pth".format(start_epoch)
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr'] = checkpoint["lr"] * 1.5
        start_epoch = checkpoint['epoch']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))
    best_epoch = 0
    for epoch in range(start_epoch, end_epoch, 1):
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        psnr = []
        psnr_train = []
        # ============Epoch Train=============== #
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            GT, LRHSI = batch['hrhsi'].cuda(), batch['lrhsi']
            up_sample = nn.functional.interpolate(LRHSI,scale_factor=4,mode="bilinear")

            optimizer.zero_grad()  # fixed
            output_HRHSI = model(LRHSI.cuda(),up_sample.cuda())
            Pixelwise_Loss = PLoss(output_HRHSI, GT)
            epoch_train_loss.append(Pixelwise_Loss.item())
            Pixelwise_Loss.backward()  # fixed
            optimizer.step()  # fixed
            psnr_train.append(PSNR_GPU(output_HRHSI, GT).item())
            if iteration % 25 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                   Pixelwise_Loss.item()))

        lr_scheduler.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        psnr_ = np.nanmean(np.array(psnr_train))
        print('Epoch: {}/{} \t training loss: {:.7f}\tpsnr:{:.2f}'.format(end_epoch, epoch, t_loss, psnr_))
        # ============Epoch Validate=============== #
        if epoch % ckpt_step == 0:
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    GT, LRHSI = batch['hrhsi'].cuda(), batch['lrhsi']
                    up_sample = nn.functional.interpolate(LRHSI, scale_factor=4,mode="bilinear")
                    output_HRHSI = model(LRHSI.cuda(),up_sample.cuda())

                    Pixelwise_Loss = PLoss(output_HRHSI, GT)
                    MyVloss = Pixelwise_Loss

                    epoch_val_loss.append(MyVloss.item())
                    psnr.append(PSNR_GPU(output_HRHSI, GT).item())
            v_loss = np.nanmean(np.array(epoch_val_loss))
            psnr = np.nanmean(np.array(psnr))
            best_epoch = epoch
            save_checkpoint(model_folder, model, optimizer, lr, epoch)

            print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
            print('             validate loss: {:.7f}'.format(v_loss))
            print('             PSNR loss: {:.7f}'.format(psnr))
    return best_epoch

# ========================Build Network==========================
train_parser = argparse.ArgumentParser(description="parser for SR network")
train_parser.add_argument("--cuda", type=int, required=False,default=1,
                      help="set it to 1 for running on GPU, 0 for CPU")
train_parser.add_argument("--epochs", type=int, default=40, help="epochs, default set to 20")
train_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
train_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
train_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
args = train_parser.parse_args()
# PaviaU
colors=103
# Pavia
# colors=102

model = SSPSR(n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=colors, n_blocks=args.n_blocks, n_feats=args.n_feats, n_scale=args.n_scale, res_scale=0.1, use_share=args.use_share, conv=default_conv)
model.to('cuda')
# ========================Dataset Setting========================
dataset_name = 'Pavia'
Method='SSPSR'
model_folder = Method + '/' + dataset_name + '/'
if not os.path.isdir(Method):
    os.mkdir(Method)


# Training Setting
Batch_size = 32
end_epoch = 2000
ckpt_step = 50
lr = 1e-4

# Resume
resume = False
start = 0


from Dataloader_tool import PaviaDataset

Train_data = PaviaDataset(f'Multispectral Image Dataset\{dataset_name}.mat',ratio=args.n_scale,type='train')
Val_data =  PaviaDataset(f'Multispectral Image Dataset\{dataset_name}.mat',ratio=args.n_scale,type='eval')


training_data_loader = DataLoader(dataset=Train_data, num_workers=0, batch_size=Batch_size, shuffle=True,
                              pin_memory=True, drop_last=False)
validate_data_loader = DataLoader(dataset=Val_data, num_workers=0, batch_size=1, shuffle=True,
                                  pin_memory=True, drop_last=True)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
bestepoch = Supervisedfusion(model,training_data_loader,validate_data_loader,model_folder=model_folder,optimizer=optimizer,lr=lr,start_epoch=start,end_epoch=end_epoch,ckpt_step=ckpt_step,RESUME=resume)

