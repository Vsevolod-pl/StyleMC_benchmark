import argparse
import torch
from argparse import Namespace
from matplotlib import pyplot as plt
from tqdm.notebook import trange

import clip
import e4e
from clip_loss import CLIPLoss
from id_loss import IDLoss
from StyleGAN import load_generator

def optimization(generator, description, latent_code_init, device='cuda', args=None):
    if args is None:
        args = {
            "lr_rampup": 0.05,
            "lr": 0.1,
            "step": 100,
            "l2_lambda": 0.005,
            "id_lambda": 0.,
            'debug': False,
            'stylegan_size': 1024,
            "ir_se50_weights": "model_ir_se50.pth"
        }
        args = Namespace(**args)

    clip_loss = CLIPLoss(args.stylegan_size, device=args.device).to(device)
    if args.id_lambda > 0:
        id_loss = IDLoss(args.ir_se50_weights).to(device)

    text_inputs = clip.tokenize(description).to(device)

    latent_code_init = latent_code_init.detach()
    latent = latent_code_init.clone().to(device)
    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=args.lr)

    clip_hist = []
    idl_hist = []
    l2_hist = []

    for i in trange(args.step):
        lr = args.lr-i*args.lr_rampup/args.step
        
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = generator([latent], input_is_latent=True, randomize_noise=False)

        c_loss = clip_loss(img_gen, text_inputs)
        if args.debug:
            clip_hist.append(c_loss.item())

        if args.id_lambda > 0:
            i_loss = id_loss(img_gen, img_orig)[0]
            if args.debug:
                idl_hist.append(i_loss.item()*args.id_lambda)
        else:
            i_loss = 0
            
        l2_loss = ((latent_code_init - latent) ** 2).sum()
        if args.debug:
            l2_hist.append(l2_loss.item()*args.l2_lambda)
        
        loss = c_loss + args.l2_lambda * l2_loss + args.id_lambda * i_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if args.debug:
        return latent.detach(), clip_hist, idl_hist, l2_hist
    return latent.detach()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Latent optimization',)
    parser.add_argument('-i', '--in-file')
    parser.add_argument('-o', '--out')
    parser.add_argument('--description')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--generator', default='./stylegan2-ffhq-config-f.pt')
    parser.add_argument('--stylegan-size', default=1024, type=int)
    parser.add_argument('--style-dim', default=512, type=int)
    parser.add_argument('--n-mlp', default=8, type=int)
    parser.add_argument('--lr-rampup', default=0.05, type=float)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--l2-lambda', default=0.005, type=float)
    parser.add_argument('--id-lambda', default=0., type=float)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--step', default=100, type=int)
    parser.add_argument('--ir-se50-weights', default='model_ir_se50.pth')
    args = parser.parse_args()

    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=args.stylegan_size, style_dim=args.style_dim, n_mlp=args.n_mlp)
    
    latent_orig = e4e.calc_latent(args.in_file, latent_avg)

    latent = optimization(generator=generator, description=args.description, latent_code_init=latent_orig, device=args.device, args=args)

    if args.debug:
        latent = latent[0]
    with torch.no_grad():
        img, _ = generator([latent], input_is_latent=True, randomize_noise=False)
    img = img[0].cpu()
    scale = img.max()-img.min()
    d = img.min()/scale
    img = img.transpose(0,2).transpose(1,0)/scale-d 
    img = img.numpy()
    plt.imsave(args.out, img)