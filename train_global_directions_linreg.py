import argparse
import torch
from argparse import Namespace
from tqdm import trange, tqdm

import clip
from StyleGAN import load_generator

stylegan_size=1024
upsample = torch.nn.Upsample(scale_factor=7)
avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

def enc_img(clip_model, img):
    return clip_model.encode_image(avg_pool(upsample(img)))

def enc_lat(generator, clip_model, latents, truncation=0.7, truncation_latent=None):
    img, _ = generator([latents], input_is_latent=True, randomize_noise=False, truncation=0.7, truncation_latent=latent_avg)
    return enc_img(clip_model, img).sum(0)


def calculate_global_directions_lin(generator, latent_avg, clip_model, num_epochs=10000, batch_size=10, lr=0.1, save_name=None, save_rate=None, device='cuda', truncation=0.7):
    rel = torch.randn(18, 512, 512).to(device)
    rel = rel.requires_grad_(True)
    b = torch.randn(512).to(device)
    b = b.requires_grad_(True)

    opt = torch.optim.Adam([rel, b], lr=lr)
    
    for epoch in trange(num_epochs):
        opt.zero_grad()
        with torch.no_grad():
            batch = torch.randn(batch_size, 512).to(device)
            img, batch, _ = generator([batch], input_is_latent=False, randomize_noise=False, truncation=truncation, truncation_latent=latent_avg, return_latents=True)
            clip_enc = enc_img(clip_model, img)
        preds = batch.view(batch_size, 18*512)@rel.view(18*512, 512)
        loss = torch.nn.functional.mse_loss(preds, clip_enc)
        loss.backward()
        opt.step()
        opt.param_groups[0]['lr'] = max(opt.param_groups[0]['lr']*0.995, 5e-4)
    
        torch.cuda.empty_cache()
        if save_name is not None and save_rate is not None and epoch % save_rate == save_rate-1:
            torch.save(rel, f'{save_name}_{epoch}.pt')
    return rel


    save_name is not None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Global directions',)
    parser.add_argument('--filename')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--generator', default='./stylegan2-ffhq-config-f.pt')
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--save-rate', default=None, type=int)
    parser.add_argument('--truncation', default=0.7, type=float)

    args = parser.parse_args()

    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    clip_model = clip_model.float()
    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=stylegan_size, style_dim=512, n_mlp=8)

    global_dirs = calculate_global_directions_lin(generator=generator, latent_avg=latent_avg,
                clip_model=clip_model, num_epochs=args.num_epochs, batch_size=args.batch_size,
                save_name=args.filename, save_rate=args.save_rate, device=args.device, lr=args.lr,
                truncation=args.truncation)

    torch.save(global_dirs, f'{args.filename}.pt')