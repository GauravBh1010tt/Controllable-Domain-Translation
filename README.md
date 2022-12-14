# Controllable-Domain-Translation
Code for our paper [Learning Style Subspaces for Controllable Unpaired Domain Translation](https://openaccess.thecvf.com/content/WACV2023/html/Bhatt_Learning_Style_Subspaces_for_Controllable_Unpaired_Domain_Translation_WACV_2023_paper.html)


### Train
`$python main.py --num_domains=2 --mode='train' --real_img_dir='data/celeb_hq/train'`

### Inference
`$python main.py --num_domains=2 --mode='sample' --real_img_dir='data/celeb_hq/train'`

## Results of Controllable domain translation

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/m2f_sub.jpg" width="700">

### Forward-backward interpolation

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/fb_int.jpg" width="700">

### Multi-attribute controllable generation

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/multi.jpg" width="500">

### Multi-domain style transfer

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/multi_dom.jpg" width="700">

### References-guided image synthesis

<img src="https://github.com/GauravBh1010tt/Controllable-Domain-Translation/blob/main/figs/figs/ref_celeb.jpg" width="700">


## BibTex:- 

@InProceedings{Bhatt_2023_WACV,
    author    = {Bhatt, Gaurav and Balasubramanian, Vineeth N.},
    title     = {Learning Style Subspaces for Controllable Unpaired Domain Translation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {4220-4229}
}
