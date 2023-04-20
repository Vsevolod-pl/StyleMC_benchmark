import argparse
import os
import torch
from matplotlib import pyplot as plt
import e4e
from StyleGAN import load_generator
from levels_mapper import load_mapper

def draw_img(img, name=None):
    with torch.no_grad():
        img = img[0].cpu()
        scale = img.max()-img.min()
        d = img.min()/scale
        img = img.transpose(0,2).transpose(1,0)/scale-d
        img = img.numpy()
        if name is not None:
            plt.imsave(name, img)
        return plt.imshow(img)

def process_img(name_in, name_out, generator, latent_avg, mapper, device='cuda'):
    with torch.no_grad():
        latent_orig = e4e.calc_latent(name_in, latent_avg, device=device).to(device)
        new_latent = latent_orig + 0.1*mapper(latent_orig)
        img_hat, _ = generator([new_latent], input_is_latent=True, randomize_noise=False)
    draw_img(img_hat, name_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Apply global directions',)
    parser.add_argument('--in-file')
    parser.add_argument('--mapper')
    parser.add_argument('-o', '--out')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--generator', default='./stylegan2-ffhq-config-f.pt')
    parser.add_argument('--stylegan-size', default=1024, type=int)
    parser.add_argument('--style-dim', default=512, type=int)
    parser.add_argument('--n-mlp', default=8, type=int)
    parser.add_argument('--debug', default=False)
    
    args = parser.parse_args()
    device=args.device
    
    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=args.stylegan_size, style_dim=args.style_dim, n_mlp=args.n_mlp)
    mapper = load_mapper(args.mapper, args.device)
    
    if os.path.isdir(args.in_file):
        for path, folder, files in os.walk(args.in_file):
            for file in files:
                process_img(f'{path}/{file}', f'{args.out}/{file}', generator, latent_avg, mapper, args.device)
    else:
        process_img(args.in_file, args.out, generator, latent_avg, mapper, args.device)