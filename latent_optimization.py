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

def optimization(generator, description, latent_code_init, device='cuda', lr_rampup=0.05, lr=0.1, num_steps=300, l2_lambda=0.008, id_lambda=0., debug=False, stylegan_size=1024, ir_se50_weights="model_ir_se50.pth"):

    clip_loss = CLIPLoss(stylegan_size, device=device).to(device)
    if id_lambda > 0:
        id_loss = IDLoss(ir_se50_weights).to(device)

    text_inputs = clip.tokenize(description).to(device)

    latent_code_init = latent_code_init.detach()
    latent = latent_code_init.clone().to(device)
    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    clip_hist = []
    idl_hist = []
    l2_hist = []

    for i in trange(num_steps):
        optimizer.param_groups[0]["lr"] = lr-i*lr_rampup/num_steps

        img_gen, _ = generator([latent], input_is_latent=True, randomize_noise=False)

        c_loss = clip_loss(img_gen, text_inputs)
        if debug:
            clip_hist.append(c_loss.item())

        if id_lambda > 0:
            i_loss = id_loss(img_gen, img_orig)[0]
            if debug:
                idl_hist.append(i_loss.item()*id_lambda)
        else:
            i_loss = 0
            
        l2_loss = ((latent_code_init - latent) ** 2).sum()
        if debug:
            l2_hist.append(l2_loss.item()*l2_lambda)
        
        loss = c_loss + l2_lambda * l2_loss + id_lambda * i_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if debug:
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
    parser.add_argument('--l2-lambda', default=0.008, type=float)
    parser.add_argument('--id-lambda', default=0., type=float)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--num-steps', default=300, type=int)
    parser.add_argument('--ir-se50-weights', default='model_ir_se50.pth')
    parser.add_argument('--stylespace', default=False, action='store_true')
    args = parser.parse_args()

    generator, latent_avg = load_generator(args.generator, device=device, stylegan_size=stylegan_size, style_dim=args.style_dim, n_mlp=args.n_mlp)
    
    latent_orig = e4e.calc_latent(args.in_file, latent_avg)

    latent = optimization(generator=generator, description=args.description,
        latent_code_init=latent_orig, device=device, lr_rampup=args.lr_rampup,
        lr=args.lr, num_steps=args.num_steps, l2_lambda=args.l2_lambda, id_lambda=args.id_lambda,
        debug=args.debug, stylegan_size=args.stylegan_size, ir_se50_weights=args.ir_se50_weights)

    if debug:
        latent = latent[0]
    with torch.no_grad():
        img, _ = generator([latent], input_is_latent=True, randomize_noise=False)
    img = img[0].cpu()
    scale = img.max()-img.min()
    d = img.min()/scale
    img = img.transpose(0,2).transpose(1,0)/scale-d 
    img = img.numpy()
    plt.imsave(args.out, img)