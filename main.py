
import os
import argparse

#from munch import Munch
from torch.backends import cudnn
import torch

#from core.data_loader import get_train_loader
#from core.data_loader import get_test_loader
from core.solver_ms import Solver
from core.dataloader import HDF5DataLoader, MSCOCO_att, AFHQ
import pickle
from torch.utils.data.sampler import SubsetRandomSampler


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    if args.mode == 'train':
        #assert len(subdirs(args.train_img_dir)) == args.num_domains
        #assert len(subdirs(args.val_img_dir)) == args.num_domains
        
        '''
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        
        
        train_loader = HDF5DataLoader(args.train_img_dir, split='train', 
                                      batch_size=args.batch_size, resize=args.img_size, latents=args.latent_dim)
        
        test_loader = HDF5DataLoader(args.train_img_dir, split='test', 
                                      batch_size=args.val_batch_size, resize=args.img_size, latents=args.latent_dim)
        
        train_it = train_loader.create_iterator_n_iters(args.total_iters)
        test_it = test_loader.create_iterator_n_iters(100)
        
        '''
        
        info = pickle.load(open(args.info_file,'rb'))
        
        train_sampler = SubsetRandomSampler(list(range(len(info['train_info']) - (len(info['train_info']) % args.batch_size) )))
        #valid_sampler = SubsetRandomSampler(list(range(len(info['val_info']))))
        
        #train_dataset = MSCOCO_att(info['train_info'], info['att_info'], resize=args.img_size)
        #val_dataset = MSCOCO_att(info['val_info'], info['att_info'], resize=args.img_size)
        
        train_dataset = AFHQ(info['train_info'][0:len(info['train_info']) - len(info['train_info']) % args.batch_size], 
                             args.par_dim, resize=args.img_size)
        val_dataset = AFHQ(info['val_info'][0:len(info['val_info']) - len(info['val_info']) % args.val_batch_size],
                           args.par_dim, resize=args.img_size)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.val_batch_size)
        
        solver.train([train_loader, val_loader])
        
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_epochs', type=int, default=100,
                        help='Number of total epochs')
    
    parser.add_argument('--val_epochs', type=int, default=10,
                        help='Number of total iterations')
    
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    
    parser.add_argument('--par_dim', type=int, default=13,
                        help='number of style subspaces: 13 for celeb and 8 for MSCOCO_attributes')
    
    parser.add_argument('--par_mode', type=str, default='concat',
                    help='concat or mul')
    
    parser.add_argument('--mask_act', type=str, default='relu',
            help='use relu, sigmoid or leaky_relu')
    
    parser.add_argument('--use_par', type=int, default=0,
                help='use partition loss or not')
    
    parser.add_argument('--lambda_par', type=float, default=1,
                help='weight for partition loss')
    
    parser.add_argument('--val_batch_size', type=int, default=20,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celebHQ_subspace_256.hdf5',
                        help='Directory containing training images')
    parser.add_argument('--info_file', type=str, default='data/AFHQ_info.pkl',
                        help='Directory containing training images')
    parser.add_argument('--real_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')
    
    parser.add_argument('--exp_name', type=str, default='demo',
                        help='Directory for saving network checkpoints')
    
    parser.add_argument('--task', type=str, default='cat2dog')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=30000)

    args = parser.parse_args()
    
    save_model_path = "expr/"+args.exp_name+'/checkpoints'
    save_img_path = "expr/"+args.exp_name+'/samples'
    save_eval_path = "expr/"+args.exp_name+'/eval'

    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_eval_path, exist_ok=True)
    
    args.sample_dir = save_img_path
    args.checkpoint_dir = save_model_path
    args.eval_dir = save_eval_path
    
    main(args)
