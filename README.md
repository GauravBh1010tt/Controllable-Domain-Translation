# Controllable-Domain-Translation
Code for our paper "Learning Style Subspaces for Controllable Unpaired Domain Translation"


### Train
`$python main.py --num_domains=2 --mode='train' --real_img_dir='data/celeb_hq/train'`

### Inference
`$python main.py --num_domains=2 --mode='sample' --real_img_dir='data/celeb_hq/train'`

## Results of Controllable domain translation

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/m2f_sub.jpg" width="900">

## Forward-backward interpolation

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/fb_int.jpg" width="900">

## Multi-attribute controllable generation

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/multi.jpg" width="900">

## Multi-domain style transfer

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/multi_dom.jpg" width="900">

## References-guided image synthesis

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/ref_celeb.jpg" width="900">