import torch
import os
# Network Tool
def save_checkpoint(model_folder, model, optimizer, lr, epoch):  # save model function

    model_out_path = model_folder + "{}.pth".format(epoch)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr":lr
    }
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    torch.save(checkpoint, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))
#  Evaluation Metrics - GPU Version
def PSNR_GPU(im_true, im_fake):
    data_range = 1
    _,C,H,W = im_true.size()
    err = torch.pow(im_true.clone()-im_fake.clone(),2).mean(dim=(-1,-2), keepdim=True)
    psnr = 10. * torch.log10((data_range**2)/err)
    return torch.mean(psnr)
def SSIM_GPU(r_img,f_img,k1=0.01, k2=0.03):
    l = 1
    x1_ = r_img.reshape(r_img.size(1),-1)
    x2_ = f_img.reshape(f_img.size(1),-1)
    u1 = x1_.mean(dim=-1,keepdim=True)
    u2 = x1_.mean(dim=-1,keepdim=True)
    Sig1 = torch.std(x1_, dim=-1,keepdim=True)
    Sig2 = torch.std(x2_, dim=-1,keepdim=True)
    sig12 = torch.sum((x1_ - u1) * (x2_ - u2), dim=-1) / (x1_.size(-1) - 1)
    c1, c2 = pow(k1 * l, 2), pow(k2 * l, 2)
    SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))
    return SSIM.mean()


