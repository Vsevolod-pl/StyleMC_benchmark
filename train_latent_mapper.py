import argparse
import torch
from argparse import Namespace
from matplotlib import pyplot as plt
from tqdm import trange

import clip
import e4e
from clip_loss import CLIPLoss
from id_loss import IDLoss
from StyleGAN import load_generator
from levels_mapper import LevelsMapper

def train(generator, latent_avg, description, mapper, num_epochs=10, batch_size=2, lr=1e-3, l2_lambda=0.8, id_lambda=0.1, device='cuda', ir_se50_weights='./model_ir_se50.pth', save_name=None, save_rate=None, debug=False, stylegan_size=1024, truncation=0.8):
    clip_hist = []
    idl_hist = []
    l2_hist = []

    opt = torch.optim.Adam(mapper.parameters(), lr=lr)
    clip_loss = CLIPLoss(stylegan_size, device=args.device).to(device)
    text_inputs = clip.tokenize(description).to(device)
    mse_loss = torch.nn.MSELoss()
    if id_lambda > 0:
        id_loss = IDLoss(ir_se50_weights).to(device)

    for epoch in trange(num_epochs):
        # for batch in tqdm(train_dataloader):
        #     # opt.param_groups[0]['lr'] *= 0.99975
        #     for step in range(args.steps):
        opt.zero_grad()
        w = torch.randn(batch_size, 512).to(device)
        with torch.no_grad():
            img_orig, w, _ = generator([w], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=truncation, truncation_latent=latent_avg)
        w_hat = w + 0.1 * mapper(w)
        img_hat, _ = generator([w_hat], input_is_latent=True, randomize_noise=False)

        cl_loss = clip_loss(img_hat, text_inputs).mean()
        clip_hist.append(cl_loss.item())

        l2 = l2_lambda*mse_loss(w_hat, w)
        l2_hist.append(l2.item())

        idloss = torch.tensor(0)
        if id_lambda > 0:
            idloss = id_loss(img_hat, img_orig)[0]*id_lambda
        idl_hist.append(idloss.item())

        loss = cl_loss + l2 + idloss
        loss.backward()
        opt.step()
        if save_name is not None and save_rate is not None and epoch % save_rate == save_rate-1:
            torch.save(mapper.state_dict(), f'{save_name}_ep={epoch+1}.pt')
    
    return mapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Latent mapper training',)
    parser.add_argument('--filename')
    parser.add_argument('--description')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--generator', default='./stylegan2-ffhq-config-f.pt')
    parser.add_argument('--stylegan-size', default=1024, type=int)
    parser.add_argument('--style-dim', default=512, type=int)
    parser.add_argument('--n-mlp', default=8, type=int)
    # parser.add_argument('--lr-rampup', default=0.05, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2-lambda', default=0.8, type=float)
    parser.add_argument('--id-lambda', default=0.1, type=float)
    parser.add_argument('--truncation', default=0.8, type=float)
    parser.add_argument('--num-epochs', default=50000, type=int)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--save-rate', default=2000, type=int)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--ir-se50-weights', default='model_ir_se50.pth')
    args = parser.parse_args()

    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=args.stylegan_size, style_dim=args.style_dim, n_mlp=args.n_mlp)
    
    mapper = LevelsMapper(args.style_dim).to(args.device)

    mapper = train(generator, latent_avg, args.description, mapper, num_epochs=args.num_epochs,
        batch_size=args.batch_size, lr=args.lr, l2_lambda=args.l2_lambda, id_lambda=args.id_lambda,
        device=args.device, ir_se50_weights=args.ir_se50_weights, save_name=args.filename,
        save_rate=args.save_rate, debug=args.debug, stylegan_size=args.stylegan_size, truncation=args.truncation)

    torch.save(mapper.state_dict(), f'{args.filename}.pt')