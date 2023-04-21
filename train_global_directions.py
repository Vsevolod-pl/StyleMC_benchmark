import argparse
import torch
from argparse import Namespace
from tqdm import trange, tqdm

from StyleGAN import load_generator

stylegan_size=1024
upsample = torch.nn.Upsample(scale_factor=7)
avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

def enc_img(clip_model, img):
    return clip_model.encode_image(avg_pool(upsample(img)))

def calculate_global_directions(generator, latent_avg, clip_model, test_step_len=0.1, num_epochs=10, batch_size=100, save_name=None, save_rate=None, device='cuda', truncation=0.7):
    rel = torch.zeros(18, 512, 512).cpu()
    dlat = torch.zeros(1, 18, 512).to(device)
    cos_sim = torch.nn.CosineSimilarity()
    for epoch in range(num_epochs):
        with torch.no_grad():
            batch = torch.randn(batch_size, 512).to(device)
            img, batch, _ = generator([batch], input_is_latent=False, randomize_noise=False, truncation=0.7, truncation_latent=latent_avg, return_latents=True)
            img_neut = enc_img(clip_model, img)
            del img
            torch.cuda.empty_cache()

            for j in trange(512):
                for i in range(18):
                    dlat *= 0
                    dlat[0, i, j] = test_step_len
                    img2, _ = generator([batch + dlat], input_is_latent=True, randomize_noise=False)
                    img_tar = enc_img(clip_model, img2)
                    # rel[i, j] += cos_sim(dt, (img_tar - img_neut))[0].cpu()
                    rel[i, j] += (img_tar - img_neut).cpu()
        if save_name is not None and save_rate is not None and epoch % save_rate == save_rate-1:
            torch.save(rel, f'{save_name}_{epoch}.pt')
    return rel


    save_name is not None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Global directions',)
    parser.add_argument('--filename')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--generator', default='./stylegan2-ffhq-config-f.pt')
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--save-rate', default=10, type=int)
    parser.add_argument('--step-size', default=0.1, type=float)
    parser.add_argument('--truncation', default=0.7, type=float)

    args = parser.parse_args()

    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=stylegan_size, style_dim=512, n_mlp=8)

    global_dirs = calculate_global_directions(generator=generator, latent_avg=latent_avg,
                clip_model=clip_model, test_step_len=args.step_size, num_epochs=args.num_epochs,
                batch_size=args.batch_size, save_name=args.filename, save_rate=args.save_rate,
                device=args.device, truncation=args.truncation)

    torch.save(global_dirs, f'{args.filename}.pt')