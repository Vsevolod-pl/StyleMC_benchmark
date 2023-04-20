import argparse
import torch
from matplotlib import pyplot as plt
from tqdm.notebook import trange

import clip
import e4e
from StyleGAN import load_generator

def draw_img(img):
    with torch.no_grad():
        img = img[0].cpu()
        scale = img.max()-img.min()
        d = img.min()/scale
        return plt.imshow(img.transpose(0,2).transpose(1,0)/scale-d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Apply global directions',)
    parser.add_argument('--in-file')
    parser.add_argument('--gd-file')
    parser.add_argument('-o', '--out')
    parser.add_argument('--description')
    parser.add_argument('--neutral', default='normal face')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--generator', default='./stylegan2-ffhq-config-f.pt')
    parser.add_argument('--stylegan-size', default=1024, type=int)
    parser.add_argument('--style-dim', default=512, type=int)
    parser.add_argument('--n-mlp', default=8, type=int)
    parser.add_argument('--alpha', default=3., type=float)
    parser.add_argument('--alpha-range', default=0., type=float)
    parser.add_argument('--alpha-points', default=1, type=int)
    parser.add_argument('--beta', default=0.6, type=float)
    parser.add_argument('--beta-max', default=None, type=float)
    parser.add_argument('--beta-points', default=1, type=int)
    parser.add_argument('--debug', default=False)
    
    args = parser.parse_args()
    device=args.device
    if args.beta_max is None:
        args.beta_max = args.beta

    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=args.stylegan_size, style_dim=args.style_dim, n_mlp=args.n_mlp)
    gd = torch.load(args.gd_file).to(device)
    
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    cos_sim2 = torch.nn.CosineSimilarity(dim=2)

    latent_orig = e4e.calc_latent(args.in_file, latent_avg)

    with torch.no_grad():
        text_neut = clip_model.encode_text(clip.tokenize(args.neutral).to(device)).cpu()
        text_tar = clip_model.encode_text(clip.tokenize(args.description).to(device)).cpu()
        dt = (text_tar-text_neut)[0].to(device)

    rel = cos_sim2(gd, dt)

    fig = plt.figure(figsize=(5*args.alpha_points, 5*args.beta_points))
    
    i = 0
    mx_it = torch.abs(rel).max().item()
    beta_start = mx_it*args.beta
    beta_end = mx_it*args.beta_max
    alpha_start = args.alpha-args.alpha_range/2
    alpha_end = args.alpha+args.alpha_range/2

    for beta in torch.linspace(beta_start, beta_end, args.beta_points):
        mask = torch.zeros_like(rel)+1
        mask[torch.abs(rel) < beta] = 0
        gamma = torch.norm(mask*rel)
        for alpha in torch.linspace(alpha_start, alpha_end, args.alpha_points):
            i += 1
            lat2 = latent_orig+(mask*rel*alpha/gamma).cuda()

            with torch.no_grad():
                img2, _ = generator([lat2], input_is_latent=True, randomize_noise=False)
            plt.subplot(args.beta_points, args.alpha_points, i)
            plt.axis('off')
            draw_img(img2)
    fig.savefig(args.out)