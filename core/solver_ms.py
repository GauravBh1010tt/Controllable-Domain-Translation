
import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims
        
        total_iters = 0

        # fetch random validation images for debugging
        #fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        #fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        train_loader, val_loader = loaders[0], loaders[1]
        
        inputs_val = next(iter(val_loader))
        # input: im_cat1, im_cat2, im_dog1, im_dog2, att_vector
        val_x_src, val_x_ref = torch.zeros_like(inputs_val[0]),torch.zeros_like(inputs_val[1])
        val_y_src, val_y_ref = [],[]

        for i in range(args.val_batch_size):
            #print (i)
            if i<5:
                val_x_src[i] = inputs_val[0][i]
                val_x_ref[i] = inputs_val[2][i]
                val_y_src.append(0)
                val_y_ref.append(1)
            elif i<10:
                val_x_src[i] = inputs_val[2][i]
                val_x_ref[i] = inputs_val[0][i]
                val_y_src.append(1)
                val_y_ref.append(0)
            elif i<15:
                val_x_src[i] = inputs_val[0][i]
                val_x_ref[i] = inputs_val[1][i]
                val_y_src.append(0)
                val_y_ref.append(0)
            else:
                val_x_src[i] = inputs_val[2][i]
                val_x_ref[i] = inputs_val[3][i]
                val_y_src.append(1)
                val_y_ref.append(1)

        val_y_src = torch.tensor(val_y_src).cuda()
        val_y_ref = torch.tensor(val_y_ref).cuda()
        val_p_vec = inputs_val[-1]
        
        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for epoch in range(args.resume_iter, args.total_epochs):
            
            for inputs in train_loader:
                # fetch images and labels
                #inputs = next(iter(train_loader))

                if np.random.random()>0.2:
                    if np.random.random()>0.5:
                        x_real, y_org = inputs[0], torch.tensor(args.batch_size*[0]).cuda()
                        x_ref, x_ref2 = inputs[2], inputs[3]
                        y_trg = torch.tensor(args.batch_size*[1]).cuda()

                    else:
                        x_real, y_org = inputs[2], torch.tensor(args.batch_size*[1]).cuda()
                        x_ref, x_ref2 = inputs[0], inputs[1]
                        y_trg = torch.tensor(args.batch_size*[0]).cuda()
                else:
                    if np.random.random()>0.5:
                        x_real, y_org = inputs[0], torch.tensor(args.batch_size*[0]).cuda()
                        x_ref, x_ref2 = inputs[1], inputs[0]
                        y_trg = torch.tensor(args.batch_size*[0]).cuda()
                    else:
                        x_real, y_org = inputs[2], torch.tensor(args.batch_size*[1]).cuda()
                        x_ref, x_ref2 = inputs[3], inputs[2]
                        y_trg = torch.tensor(args.batch_size*[1]).cuda()

                z_trg, z_trg2 = torch.randn(args.batch_size, args.latent_dim).cuda(), torch.randn(args.batch_size, args.latent_dim).cuda()
                p_vec = inputs[-1]

                #print (p_vec.shape)
                #bre

                masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

                # train the discriminator
                d_loss, d_losses_latent = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks, par_vec=p_vec)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()

                d_loss, d_losses_ref = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks, par_vec=p_vec)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()

                # train the generator
                g_loss, g_losses_latent = compute_g_loss(
                    nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks, par_vec=p_vec)
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                optims.mapping_network.step()
                optims.style_encoder.step()

                g_loss, g_losses_ref = compute_g_loss(
                    nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks, par_vec=p_vec)
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()

                # compute moving average of network parameters
                moving_average(nets.generator, nets_ema.generator, beta=0.999)
                moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
                moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

                # decay weight for diversity sensitive loss
                if args.lambda_ds > 0:
                    args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

                # print out log info
                if (total_iters+1) % args.print_every == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                    log = "Elapsed time [%s], Iteration [%i/%i], Model [%s], " % (elapsed, total_iters+1, args.total_epochs*len(train_loader), args.exp_name)
                    all_losses = dict()
                    for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                    all_losses['G/lambda_ds'] = args.lambda_ds
                    log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                    print(log)

                # generate images for debugging
                if (total_iters+1) % args.sample_every == 0:
                    os.makedirs(args.sample_dir, exist_ok=True)

                    utils.debug_image(nets_ema, args, val_x_src, val_x_ref, val_y_src, val_y_ref, val_p_vec, step=total_iters+1)

                # save model checkpoints
                if (total_iters+1) % args.save_every == 0:
                    self._save_checkpoint(step=i+1)

                # compute FID and LPIPS if necessary
                if (total_iters+1) % args.eval_every == 0:
                    calculate_metrics(nets_ema, args, i+1, mode='latent', test_it = val_loader)
                    calculate_metrics(nets_ema, args, i+1, mode='reference', test_it = val_loader)
                    
                total_iters += 1

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, par_vec=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    
    #print (out)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
            
        if args.par_dim>0:
            if args.par_mode == 'concat':
                #print (s_trg, par_vec)
                s_trg = torch.cat([s_trg.float(), par_vec.float()], -1)
            else:
                masked_par_vec = nets.mask_mul(par_vec.float())
                s_trg = torch.cat([s_trg.float(), masked_par_vec.float()], -1)
            
            x_fake, _ = nets.generator(x_real, s_trg, masks=masks)
        else:
            x_fake = nets.generator(x_real, s_trg, masks=masks)
            
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, par_vec=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)
    
    s_trg_org = s_trg
    
    if args.par_dim>0:
        if args.par_mode == 'concat':
            s_trg = torch.cat([s_trg.float(), par_vec.float()], -1)
        else:
            masked_par_vec = nets.mask_mul(par_vec.float())
            s_trg = torch.cat([s_trg.float(), masked_par_vec.float()], -1)
        
        x_fake, out_par_src = nets.generator(x_real, s_trg, masks=masks)
    else:
        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg_org))
    

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
        
    if args.par_dim>0:
        
        if args.par_mode == 'concat':
            s_trg2 = torch.cat([s_trg2.float(), par_vec.float()], -1)
        else:
            masked_par_vec = nets.mask_mul(par_vec.float())
            s_trg2 = torch.cat([s_trg2.float(), masked_par_vec.float()], -1)

        x_fake2, out_par_ref = nets.generator(x_real, s_trg2, masks=masks)
    else:
        x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
        
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    
    
    # partition loss
    loss_fun = torch.nn.MSELoss()
    
    #loss_par = 0.5*(torch.mean(torch.abs(out_par_src - par_vec)) + torch.mean(torch.abs(out_par_ref - par_vec)))
    
    if args.par_dim>0:
        loss_par = 0.5*(loss_fun(out_par_src.double(), par_vec.double()) + loss_fun(out_par_ref.double(), par_vec.double()))
    
    #print ('loss_par', loss_par)
    #print (par_vec[0])
    #print (out_par_src[0])
    #print (out_par_ref[0])

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    
    if args.par_dim>0:
        if args.par_mode == 'concat':
            s_org = torch.cat([s_org.float(), par_vec.float()], -1)
        else:
            masked_par_vec = nets.mask_mul(par_vec.float())
            s_org = torch.cat([s_org.float(), masked_par_vec.float()], -1)

        x_rec, _ = nets.generator(x_fake, s_org, masks=masks)
    else:
        x_rec = nets.generator(x_fake, s_org, masks=masks)
        
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    
    if args.par_dim>0:
        loss += args.lambda_par*loss_par
        
        return loss, Munch(adv=loss_adv.item(),
                           sty=loss_sty.item(),
                           ds=loss_ds.item(),
                           cyc=loss_cyc.item(),
                             par = loss_par.item())
    else:
        return loss, Munch(adv=loss_adv.item(),
                   sty=loss_sty.item(),
                   ds=loss_ds.item(),
                   cyc=loss_cyc.item())
                           

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.reshape(batch_size, -1).sum(1).mean(0)
    return reg
