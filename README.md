# CP-Mamba: Controlled Propagation for Robust Few-Shot Segmentation with State Space Models

This repository contains the code for our paper "*CP-Mamba: Controlled Propagation for Robust Few-Shot Segmentation with State Space Models*".

> **Abstract**: *Few-Shot Segmentation (FSS) aims to segment novel objects in query images given only a few annotated support samples. While State Space Models like Mamba have recently emerged as efficient alternatives to Transformers for long-range modeling, their application to FSS remains challenging. Existing SSM-based methods suffer from recurrent contamination, where the uniform injection of support information into the hidden state accumulates irrelevant cues, leading to semantic drift and error propagation in deeper layers. Furthermore, the entanglement of support-query interactions with scan-boundary artifacts prevents effective noise isolation. To address these limitations, we propose Controlled Propagation Mamba (CP-Mamba), a unified framework that achieves robust cross-image alignment through Controlled Propagation. Our approach regulates the process along two dimensions: (1) Representation Control: We introduce a Dynamic Prototype Module for query-conditioned prototype refinement and a Multi-Scale Frequency Module to decouple structural semantics from texture noise in the frequency domain, replacing static support representations. (2) Fusion Control: We propose an Adaptive Mix-Mamba mechanism featuring a similarity-conditioned support recap strategy to selectively modulate injection intensity, alongside register tokens that act as buffers to stabilize state evolution and absorb boundary noise. Extensive experiments on PASCAL-5$^i$ and COCO-20$^i$ demonstrate that CP-Mamba can surpass existing state-of-the-arts by up to 0.8\% and 1.1\% in mIoU, respectively, while maintaining linear complexity.*

## Dependencies

- Python 3.10
- PyTorch 1.12.0
- cuda 11.6
- torchvision 0.13.0
```
> conda env create -f env.yaml
```

## Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

You can download the pre-processed PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> datasets [here](https://entuedu-my.sharepoint.com/:f:/g/personal/qianxion001_e_ntu_edu_sg/ErEg1GJF6ldCt1vh00MLYYwBapLiCIbd-VgbPAgCjBb_TQ?e=ibJ4DM), and extract them into `data/` folder. Then, you need to create a symbolic link to the `pascal/VOCdevkit` data folder as follows:
```
> ln -s <absolute_path>/data/pascal/VOCdevkit <absolute_path>/data/VOCdevkit2012
```

The directory structure is:

    ../
    ├── HMNet/
    └── data/
        ├── VOCdevkit2012/
        │   └── VOC2012/
        │       ├── JPEGImages/
        │       ├── ...
        │       └── SegmentationClassAug/
        └── MSCOCO2014/           
            ├── annotations/
            │   ├── train2014/ 
            │   └── val2014/
            ├── train2014/
            └── val2014/

## Training
- **Commands**:
  ```
  sh train_pascal_proto.sh {Port} {Split: 0/1/2/3} {Net: resnet50/vgg} {Postfix: manet/manet_5s}
  sh train_coco_proto.sh {Port} {Split: 0/1/2/3} {Net: resnet50/vgg} {Postfix: manet/manet_5s}

  # e.g.
  sh train_pascal_proto.sh 8888 0 resnet50 manet
  sh train_coco_proto.sh 8888 0 resnet50 manet_5s
  ```


## Testing

- **Commands**:
  ```
  sh test_pascal_proto.sh {Split: 0/1/2/3} {Net: resnet50/vgg} {Postfix: manet/manet_5s}
  sh test_coco_proto.sh {Split: 0/1/2/3} {Net: resnet50/vgg} {Postfix: manet/manet_5s}

  # e.g.
  sh test_pascal_proto.sh 0 resnet50 manet
  sh test_coco_proto.sh 0 resnet50 manet_5s
  ```

## References

This repo is mainly built based on [BAM](https://github.com/chunbolang/BAM) and [HMNet](https://github.com/Sam1224/HMNet). Thanks for their great works!
