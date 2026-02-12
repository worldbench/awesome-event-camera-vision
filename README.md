[![Awesome Logo](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![arXiv](https://img.shields.io/badge/arXiv-260x.xxxxx-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/260x.xxxxx)
![Visitors](https://komarev.com/ghpvc/?username=worldbench&repo=awesome-event-camera-vision&label=Hello,%20Visitor%20&color=yellow&style=social)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-red.svg?style=flat)](https://github.com/worldbench/awesome-event-camera-vision/pulls)

# :sunglasses: Awesome Event Camera Vision

This survey ...


For more details, kindly refer to our [paper](https://huggingface.co/papers/260x.xxxxx) and [project page](https://worldbench.github.io/awesome-event-camera-vision). :rocket:


### :books: Citation 

If you find this work helpful for your research, please kindly consider citing our papers:
```bib
@article{survey_event_camera,
    title   = {Event Camera Vision in the Era of Large Models: A Survey},
    author  = {Lingdong Kong and Haiqian Han and Lai Xing Ng and Xiangyang Ji and Wei Tsang Ooi and Benoit R. Cottereau},
    journal = {arXiv preprint arXiv:260x.xxxxx},
    year    = {2026}
}
```


### Table of Contents
- [**0. Background**](#background)
- [**1. Benchmarks \& Datasets**](#1-benchmarks--datasets)
    - [Benchmarks](#benchmarks)
    - [Workshops](#workshops)
    - [Datasets](#datasets)
- [**2. Event Camera Perception**](#2-event-camera-perception)
    -  [Event Camera Object Detection](#one-event-camera-object-detection)
    -  [Event Camera Semantic Segmentation](#two-event-camera-semantic-segmentation)
    -  [Event Camera Depth Estimation](#three-event-camera-depth-estimation)
    -  ...
    -  ...
- [**3. Event Camera Reconstruction**](#3-event-camera-reconstruction)
    - [Datasets for Reconstruction](#datasets-for-reconstruction)
    - [Event-based 2D Reconstruction](#event-based-2d-reconstruction)
        - [Discriminative Reconstruction Models](#discriminative-reconstruction-models)
        - [Generative Reconstruction Models](#generative-reconstruction-models)
        - [Self-Supervised & Pre-training Frameworks](#self-supervised--pre-training-frameworks)
    - [Event-based 3D Reconstruction](#event-based-3d-reconstruction)
        - [Geometric & Semi-Dense Reconstruction](#geometric--semi-dense-reconstruction)
        - [Neural Radiance Fields (NeRF) with Events](#neural-radiance-fields-nerf-with-events)
        - [3D Gaussian Splatting (3DGS) with Events](#3d-gaussian-splatting-3dgs-with-events)
        - [Generalizable 3D Reconstruction](#generalizable-3d-reconstruction)
- [**4. Event Camera Understanding**](#4-event-camera-understanding)
    -  ...
    -  ...
    -  ...
- [**5. Applications**](#5-applications)
    -  ...
    -  ...
    -  ...
- [**6. Other Resources**](#6-other-resources)
    - [Tutorials](#tutorials)
    - [Talks \& Seminars](#talks--seminars)
    - [Relevant Surveys](#relevant-surveys)
- [**7. Acknowledgements**](#7-acknowledgements)



# Background

...




# 1. Benchmarks & Datasets

### Benchmarks

| <img width="125px" src="docs/figures/talk2event.png"> | <img width="125px" src="docs/figures/e-deflare.png"> | <img width="125px" src="docs/figures/eventfly.png"> |
|:-:|:-:|:-:|
| [**Talk2Event**](https://talk2event.github.io/) | [**E-Deflare**](https://e-flare.github.io/) | [**EventFly**](https://event-fly.github.io/) | 
| [...]() |




### Workshops

| Theme | Venue | Date | Location | Recording |
|:-|:-:|:-:|:-:|:-:|
| [Workshop on Neuromorphic Perception for Real World Robotics (NeuRobots)](https://sites.google.com/view/neurobots2025) | IROS 2025 | October 24, 2025 | Hangzhou | [[YouTube](https://www.youtube.com/playlist?list=PL41Hj1v8NO3NjD6K8s-GAmypvUM41VpSS)] |
| [Workshop on Event-Based Vision](https://eventvision-robotics.github.io/iros_workshop/) | IROS 2025 | October 20, 2025 | Hangzhou | - |
| [The 2nd Workshop on Neuromorphic Vision (NeVi)](https://sites.google.com/view/nevi-2025/home-page) | ICCV 2025 | October 20, 2025 | Honolulu | - |
| [The 5th International Workshop on Event-Based Vision](https://tub-rip.github.io/eventvision2025/) | CVPR 2025 | June 12, 2025 | Nashville | - |
| [The 4th International Workshop on Event-Based Vision](https://tub-rip.github.io/eventvision2023/) | CVPR 2023 | June 19, 2023 | Vancouver | [[YouTube](https://www.youtube.com/playlist?list=PLeXWz-g2If96iotpzgBNNTr9VA6hG-LLK)] |
| [The 3rd International Workshop on Event-Based Vision](https://tub-rip.github.io/eventvision2021/) | CVPR 2021 | June 19, 2021 | Virtual | [[YouTube](https://www.youtube.com/playlist?list=PLeXWz-g2If95mjNpA-y-WIoDaoB8WtmE7)] |
| [The 2nd International Workshop on Event-Based Vision](https://rpg.ifi.uzh.ch/CVPR19_event_vision_workshop.html) | CVPR 2019 | June 17, 2019 | Long Beach | [[YouTube](https://www.youtube.com/playlist?list=PLeXWz-g2If97iGiuBHmnW8IFIxwvSeCHx)] |
| [The 1st International Workshop on Event-Based Vision](https://rpg.ifi.uzh.ch/ICRA17_event_vision_workshop.html) | ICRA 2017 | June 2, 2017 | Singapore | [[YouTube](https://www.youtube.com/playlist?list=PLeXWz-g2If94k8mw6GcKU5C9PUgM1sK0U)] |


### Datasets

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
||
| `DSEC` | [![arXiv](https://img.shields.io/badge/arXiv-2103.06011-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2103.06011)<br>DSEC: A Stereo Event Camera Dataset for Driving Scenarios | RA-L 2021 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://dsec.ifi.uzh.ch/) | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/DSEC)](https://github.com/uzh-rpg/DSEC) |
| `DSEC-Semantic` |  | ECCV 2024 |  | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/ess)](https://github.com/uzh-rpg/ess) |
|  |
|  |


# 2. Event Camera Perception

### :one: Event Camera Object Detection

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub | 
|:-:|:-|:-:|:-:|:-:|
| `DVS-Detection` | [![arXiv](https://img.shields.io/badge/arXiv-1709.09323-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1709.09323)<br>Pseudo-Labels for Supervised Learning on Dynamic Vision Sensor Data, Applied to Object Detection under Ego-Motion | CVPRW 2018 | - | - |
| `RENet` | [![arXiv](https://img.shields.io/badge/arXiv-2209.08323-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2209.08323)<br>RGB-Event Fusion for Moving Object Detection in Autonomous Driving | ICRA 2023 | - | [![GitHub](https://img.shields.io/github/stars/ZZY-Zhou/RENet)](https://github.com/ZZY-Zhou/RENet) |
| `RVT` | [![arXiv](https://img.shields.io/badge/arXiv-2212.05598-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2212.05598)<br>Recurrent Vision Transformers for Object Detection with Event Cameras | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/RVT)](https://github.com/uzh-rpg/RVT) |
|  |
|  |



### :two: Event Camera Semantic Segmentation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub | 
|:-:|:-|:-:|:-:|:-:|
||
|  |
|  |



### :three: Event Camera Depth Estimation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub | 
|:-:|:-|:-:|:-:|:-:|
||
|  |
|  |



# 3. Event Camera Reconstruction

## Datasets for Reconstruction

> :timer_clock: In chronological order, from the earliest to the latest.

### 2D Paired Datasets

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `HQF` | [![arXiv](https://img.shields.io/badge/arXiv-2003.09078-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2003.09078)<br>Reducing the Sim-to-Real Gap for Event Cameras | ECCV 2020 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://timostoff.github.io/20ecnn) | - |
| `RLED` | [Event Camera Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Seeing_Motion_at_Nighttime_with_an_Event_Camera_CVPR_2024_paper.pdf)<br>Seeing Motion at Nighttime with an Event Camera | CVPR 2024 | - | [![GitHub](https://img.shields.io/github/stars/Liu-haoyue/NER-Net)](https://github.com/Liu-haoyue/NER-Net) |
| `EvLight` | [Event Camera Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Towards_Robust_Event-guided_Low-Light_Image_Enhancement_A_Large-Scale_Real-World_Event-Image_CVPR_2024_paper.html)<br>Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach | CVPR 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlislab22.github.io/eg-lowlight/) | [![GitHub](https://img.shields.io/github/stars/EthanLiang99/EvLight)](https://github.com/EthanLiang99/EvLight) |
| `RELED` | [Event Camera Paper](https://link.springer.com/chapter/10.1007/978-3-031-73254-6_25)<br>Towards Real-World Event-Guided Low-Light Video Enhancement and Deblurring | ECCV 2024 | - | [![GitHub](https://img.shields.io/github/stars/intelpro/ELEDNet)](https://github.com/intelpro/ELEDNet) |
| `EventAID` | [![arXiv](https://img.shields.io/badge/arXiv-2312.08220-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/pdf/2312.08220)<br>EventAid: Benchmarking Event-aided Image/Video Enhancement Algorithms with Real-captured Hybrid Dataset | TPAMI 2025 | - | - |
|  |

### 3D & Multi-View Datasets

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `IJRR` | [![arXiv](https://img.shields.io/badge/arXiv-1610.08336-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1610.08336)<br>The Event-Camera Dataset and Simulator: Event-based Data for Pose Estimation, Visual Odometry, and SLAM | IJRR 2017 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://rpg.ifi.uzh.ch/davis_data.html) | - |
| `MVSEC` | [![arXiv](https://img.shields.io/badge/arXiv-1801.10202-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1801.10202)<br>The Multivehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception | RA-L 2018 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://daniilidis-group.github.io/mvsec/) | - |
| `CED` | [![arXiv](https://img.shields.io/badge/arXiv-1904.10772-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/pdf/1904.10772)<br>CED: Color Event Camera Dataset | CVPRW 2019 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://rpg.ifi.uzh.ch/CED.html) | - |
| `TUM-VIE` | [![arXiv](https://img.shields.io/badge/arXiv-2108.07329-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2108.07329)<br>TUM-VIE: The TUM Stereo Visual-Inertial Event Dataset | IROS 2021 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset) | - |
| `EDS` | [![arXiv](https://img.shields.io/badge/arXiv-2204.07640-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2204.07640)<br>Event-Aided Direct Sparse Odometry | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/eds-buildconf)](https://github.com/uzh-rpg/eds-buildconf) |
| `VECtor` | [Event Camera Paper](https://ieeexplore.ieee.org/document/9809788)<br>VECtor: A Versatile Event-Centric Benchmark for Multi-Sensor SLAM | RA-L 2022 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://star-datasets.github.io/vector/) | - |
| `PAEv3d` | [![arXiv](https://img.shields.io/badge/arXiv-2401.17121-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2401.17121)<br>Physical Priors Augmented Event-Based 3D Reconstruction | ICRA 2024 | - | [![GitHub](https://img.shields.io/github/stars/Mercerai/PAEv3d)](https://github.com/Mercerai/PAEv3d) |
|  |


## Event-based 2D Reconstruction

### Discriminative Reconstruction Models

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `E2VID` | [![arXiv](https://img.shields.io/badge/arXiv-1904.08298-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1904.08298)<br>Events-to-Video: Bringing Modern Computer Vision to Event Cameras | CVPR 2019 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_e2vid)](https://github.com/uzh-rpg/rpg_e2vid) |
| `mEDI` | [IEEE Paper](https://ieeexplore.ieee.org/abstract/document/9252186)<br>Bringing a Blurry Frame Alive at High Frame-Rate with an Event Camera | CVPR 2019 | - | [![GitHub](https://img.shields.io/github/stars/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera)](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera) |
| `E2VID+` | [![arXiv](https://img.shields.io/badge/arXiv-2003.09078-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2003.09078)<br>Reducing the Sim-to-Real Gap for Event Cameras | ECCV 2020 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://timostoff.github.io/20ecnn) | [![GitHub](https://img.shields.io/github/stars/TimoStoff/event_cnn_minimal)](https://github.com/TimoStoff/event_cnn_minimal) |
| `eSL-Net` | [ECCV Paper](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_10)<br>Event Enhanced High-Quality Image Recovery | ECCV 2020 | - | [![GitHub](https://img.shields.io/github/stars/ShinyWang33/eSL-Net)](https://github.com/ShinyWang33/eSL-Net) |
| `SPADE-E2VID` | [IEEE Paper](https://ieeexplore.ieee.org/abstract/document/9337171)<br>SPADE-E2VID: Spatially-Adaptive Denormalization for Event-Based Video Reconstruction | TIP 2021 | - | [![GitHub](https://img.shields.io/github/stars/RodrigoGantier/SPADE_E2VID)](https://github.com/RodrigoGantier/SPADE_E2VID) |
| `TimeLens` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Tulyakov_Time_Lens_Event-Based_Video_Frame_Interpolation_CVPR_2021_paper.html)<br>TimeLens: Event-based Video Frame Interpolation | CVPR 2021 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://rpg.ifi.uzh.ch/TimeLens.html) | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_timelens)](https://github.com/uzh-rpg/rpg_timelens) |
| `ET-Net` | [ICCV Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Weng_Event-Based_Video_Reconstruction_Using_Transformer_ICCV_2021_paper.pdf)<br>Event-based Video Reconstruction Using Transformer | ICCV 2021 | - | [![GitHub](https://img.shields.io/github/stars/warranweng/et-net)](https://github.com/warranweng/et-net) |
| `E-CIR` | [![arXiv](https://img.shields.io/badge/arXiv-2203.01935-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2203.01935)<br>E-CIR: Event-Enhanced Continuous Intensity Recovery | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/chensong1995/E-CIR)](https://github.com/chensong1995/E-CIR) |
| `STIR` | [![arXiv](https://img.shields.io/badge/arXiv-2303.13767-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2303.13767)<br>Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution | CVPR 2022 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlis2022.github.io/cvpr23/egvsr) | [![GitHub](https://img.shields.io/github/stars/yunfanLu/INR-Event-VSR)](https://github.com/yunfanLu/INR-Event-VSR) |
| `EVSNN` | [![arXiv](https://img.shields.io/badge/arXiv-2201.10943-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2201.10943)<br>Event-based Video Reconstruction via Potential-assisted Spiking Neural Network | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/LinZhu111/EVSNN)](https://github.com/LinZhu111/EVSNN) |
| `EFNet` | [ECCV Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780403.pdf)<br>Event-Based Fusion for Motion Deblurring with Cross-modal Attention | ECCV 2022 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://ahupujr.github.io/EFNet/) | [![GitHub](https://img.shields.io/github/stars/AHupuJR/EFNet)](https://github.com/AHupuJR/EFNet) |
| `SuperFast` | [IEEE Paper](https://ieeexplore.ieee.org/document/9962797)<br>SuperFast: 200× Video Frame Interpolation via Event Camera | TPAMI 2023 | - | [![GitHub](https://img.shields.io/github/stars/lisiqi19971013/SuperFast)](https://github.com/lisiqi19971013/SuperFast) |
| `eSL-Net++` | [![arXiv](https://img.shields.io/badge/arXiv-2302.13766-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2302.13766)<br>Learning to Super-Resolve Blurry Images with Events | TPAMI 2023 | - | [![GitHub](https://img.shields.io/github/stars/gistvision/e2sri)](https://github.com/gistvision/e2sri) |
| `E-SAI` | [TPAMI Paper](https://ieeexplore.ieee.org/document/9740355)<br>Learning to See Through with Events | TPAMI 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://dvs-whu.cn/projects/esai/) | [![GitHub](https://img.shields.io/github/stars/dvs-whu/E-SAI)](https://github.com/dvs-whu/E-SAI) |
| `REFID` | [![arXiv](https://img.shields.io/badge/arXiv-2301.05191-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2301.05191)<br>Event-Based Frame Interpolation with Ad-hoc Deblurring | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/AHupuJR/REFID)](https://github.com/AHupuJR/REFID) |
| `CBMNet` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Event-Based_Video_Frame_Interpolation_With_Cross-Modal_Asymmetric_Bidirectional_Motion_Fields_CVPR_2023_paper.html)<br>Event-Based Video Frame Interpolation With Cross-Modal Asymmetric Bidirectional Motion Fields | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/intelpro/CBMNet)](https://github.com/intelpro/CBMNet) |
| `Revisit-EBVFI` | [![arXiv](https://img.shields.io/badge/arXiv-2307.12558-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2307.12558)<br>Revisiting Event-Based Video Frame Interpolation | IROS 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://jiabenchen.github.io/revisit_event) | - |
| `NeurImg-HDR` | [Google Drive](https://drive.google.com/file/d/1gfdc9axSIHO3OOIXL6yTCGljj1l54gqO/view)<br>Hybrid High Dynamic Range Imaging Fusing Neuromorphic and Conventional Images | TPAMI 2023 | - | [![GitHub](https://img.shields.io/github/stars/hjynwa/NeurImg-HDR)](https://github.com/hjynwa/NeurImg-HDR) |
| `HDRev-Net` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Learning_Event_Guided_High_Dynamic_Range_Video_Reconstruction_CVPR_2023_paper.html)<br>Learning Event Guided High Dynamic Range Video Reconstruction | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/YixinYang-00/HDRev)](https://github.com/YixinYang-00/HDRev) |
| `EvLowLight` | [ICCV Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liang_Coherent_Event_Guided_Low-Light_Video_Enhancement_ICCV_2023_paper.html)<br>Coherent Event Guided Low-Light Video Enhancement | ICCV 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://sherrycattt.github.io/EvLowLight/) | [![GitHub](https://img.shields.io/github/stars/sherrycattt/EvLowLight)](https://github.com/sherrycattt/EvLowLight) |
| `TimeLens-XL` | [ECCV Paper](https://link.springer.com/chapter/10.1007/978-3-031-73226-3_11)<br>TimeLens-XL: Real-Time Event-Based Video Frame Interpolation with Large Motion | ECCV 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://openimaginglab.github.io/TimeLens-XL/) | [![GitHub](https://img.shields.io/github/stars/OpenImagingLab/TimeLens-XL)](https://github.com/OpenImagingLab/TimeLens-XL) |
| `DA-Deblur` | [ECCV Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06299.pdf)<br>Motion Aware Event Representation-Driven Image Deblurring | ECCV 2024 | - | [![GitHub](https://img.shields.io/github/stars/ZhijingS/DA_event_deblur)](https://github.com/ZhijingS/DA_event_deblur) |
| `UniINR` | [![arXiv](https://img.shields.io/badge/arXiv-2305.15078-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2305.15078)<br>UniINR: Event-Guided Unified Rolling Shutter Correction, Deblurring, and Interpolation | ECCV 2024 | - | [![GitHub](https://img.shields.io/github/stars/yunfanLu/UniINR)](https://github.com/yunfanLu/UniINR) |
| `HyperE2VID` | [![arXiv](https://img.shields.io/badge/arXiv-2305.06382-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2305.06382)<br>HyperE2VID: Improving Event-Based Video Reconstruction via Hypernetworks | TIP 2024 | - | [![GitHub](https://img.shields.io/github/stars/ercanburak/HyperE2VID)](https://github.com/ercanburak/HyperE2VID) |
| `NER-Net` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Seeing_Motion_at_Nighttime_with_an_Event_Camera_CVPR_2024_paper.html)<br>Seeing Motion at Nighttime with an Event Camera | CVPR 2024 | - | [![GitHub](https://img.shields.io/github/stars/Liu-haoyue/NER-Net)](https://github.com/Liu-haoyue/NER-Net) |
| `EVFI-DS` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Video_Frame_Interpolation_via_Direct_Synthesis_with_the_Event-based_Reference_CVPR_2024_paper.html)<br>Video Frame Interpolation via Direct Synthesis with the Event-based Reference | CVPR 2024 | - | - |
| `EvLight` | [![arXiv](https://img.shields.io/badge/arXiv-2404.00834-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2404.00834)<br>Towards Robust Event-Guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach | CVPR 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlislab22.github.io/eg-lowlight/) | [![GitHub](https://img.shields.io/github/stars/EthanLiang99/EvLight)](https://github.com/EthanLiang99/EvLight) |
| `Noise2Image` | [![arXiv](https://img.shields.io/badge/arXiv-2404.01298-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2404.01298)<br>Noise2Image: Noise-Enabled Static Scene Recovery for Event Cameras | Optica 2025 | - | [![GitHub](https://img.shields.io/github/stars/rmcao/noise2image)](https://github.com/rmcao/noise2image) |
| `EvTexture` | [![arXiv](https://img.shields.io/badge/arXiv-2406.13457-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.13457)<br>EvTexture: Event-driven Texture Enhancement for Video Super-Resolution | ICML 2024 | - | [![GitHub](https://img.shields.io/github/stars/DachunKai/EvTexture)](https://github.com/DachunKai/EvTexture) |
| `ELEDNet` | [ECCV Paper](https://link.springer.com/chapter/10.1007/978-3-031-73254-6_25)<br>Towards Real-World Event-Guided Low-Light Video Enhancement and Deblurring | ECCV 2024 | - | [![GitHub](https://img.shields.io/github/stars/intelpro/ELEDNet)](https://github.com/intelpro/ELEDNet) |
| `STLR` | [ECCV Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05843.pdf)<br>Spike-temporal Latent Representation for Energy-Efficient Event-to-Video Reconstruction | ECCV 2024 | - | - |
| `MamEVSR` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_Event-based_Video_Super-Resolution_via_State_Space_Models_CVPR_2025_paper.pdf)<br>Event-based Video Super-Resolution via State Space Models | CVPR 2025 | - | - |
| `EvLight++` | [IEEE Paper](https://ieeexplore.ieee.org/document/11192751)<br>EvLight++: Low-Light Video Enhancement With an Event Camera: A Large-Scale Real-World Dataset, Novel Method, and More | TPAMI 2026 | - | [![GitHub](https://img.shields.io/github/stars/EthanLiang99/EvLight)](https://github.com/EthanLiang99/EvLight) |
|  |


### Generative Reconstruction Models

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `cGAN-E2V` | [CVPR Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Event-Based_High_Dynamic_Range_Image_and_Very_High_Frame_Rate_CVPR_2019_paper.pdf)<br>Event-Based High Dynamic Range Image and Very High Frame Rate Video Generation Using Conditional Generative Adversarial Networks | CVPR 2019 | - | - |
| `E2VIDiff` | [![arXiv](https://img.shields.io/badge/arXiv-2407.08231-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2407.08231)<br>E2VIDiff: Perceptual Events-to-Video Reconstruction using Diffusion Priors | arXiv 2024 | - | - |
| `TRG-Diffusion` | [ECCV Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05690.pdf)<br>Temporal Residual Guided Diffusion Framework for Event-Driven Video Reconstruction | ECCV 2024 | - | - |
| `REVDM` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Repurposing_Pre-trained_Video_Diffusion_Models_for_Event-based_Video_Interpolation_CVPR_2025_paper.html)<br>Repurposing Pre-trained Video Diffusion Models for Event-based Video Interpolation | CVPR 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vdm-evfi.github.io/) | [![GitHub](https://img.shields.io/github/stars/codingrex/VDM_EVFI)](https://github.com/codingrex/VDM_EVFI) |
| `EGDeblurring` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Diffusion-based_Event_Generation_for_High-Quality_Image_Deblurring_CVPR_2025_paper.html)<br>Diffusion-based Event Generation for High-Quality Image Deblurring | CVPR 2025 | - | [![GitHub](https://img.shields.io/github/stars/XinanXie/EGDeblurring)](https://github.com/XinanXie/EGDeblurring) |
|  |


### Self-Supervised & Pre-training Frameworks

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| - | [ECCV Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630647.pdf)<br>Learning to See in the Dark with Events | ECCV 2020 | - | - |
| - | [![arXiv](https://img.shields.io/badge/arXiv-2009.08283-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2009.08283)<br>Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy | CVPR 2021 | - | - |
| `TimeReplayer` | [![arXiv](https://img.shields.io/badge/arXiv-2203.13859-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2203.13859)<br>TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation | CVPR 2022 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://sites.google.com/view/timereplayer) | - |
| `EVDI` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Unifying_Motion_Deblurring_and_Frame_Interpolation_With_Events_CVPR_2022_paper.pdf)<br>Unifying Motion Deblurring and Frame Interpolation with Events | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/XiangZ-0/EVDI)](https://github.com/XiangZ-0/EVDI) |
| - | [![arXiv](https://img.shields.io/badge/arXiv-2301.01928-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2301.01928)<br>Event Camera Data Pre-training | ICCV 2023 | - | [![GitHub](https://img.shields.io/github/stars/Yan98/Event-Camera-Data-Pre-training)](https://github.com/Yan98/Event-Camera-Data-Pre-training) |
| `SAN` | [![arXiv](https://img.shields.io/badge/arXiv-2308.05932-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2308.05932)<br>Generalizing Event-Based Motion Deblurring in Real-World Scenarios | ICCV 2023 | - | [![GitHub](https://img.shields.io/github/stars/xiangz-0/gem)](https://github.com/xiangz-0/gem) |
| `EvDeraining` | [ICCV Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Unsupervised_Video_Deraining_with_An_Event_Camera_ICCV_2023_paper.pdf)<br>Unsupervised Video Deraining with an Event Camera | ICCV 2023 | - | [![GitHub](https://img.shields.io/github/stars/booker-max/Unsupervised-Deraining-with-Event-Camera)](https://github.com/booker-max/Unsupervised-Deraining-with-Event-Camera) |
| `MEM` | [![arXiv](https://img.shields.io/badge/arXiv-2212.10368-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2212.10368)<br>Masked Event Modeling: Self-Supervised Pretraining for Event Cameras | WACV 2024 | - | [![GitHub](https://img.shields.io/github/stars/tum-vision/mem)](https://github.com/tum-vision/mem) |
| `EvINR` | [![arXiv](https://img.shields.io/badge/arXiv-2407.18500-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2407.18500)<br>Revisit Event Generation Model: Self-Supervised Learning of Event-to-Video Reconstruction with Implicit Neural Representations | ECCV 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlislab22.github.io/EvINR/) | [![GitHub](https://img.shields.io/github/stars/wzpscott/EvINR)](https://github.com/wzpscott/EvINR) |
| `Sim2Real-EVFI` | [![arXiv](https://img.shields.io/badge/arXiv-2406.08090-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.08090)<br>From Sim-to-Real: Toward General Event-Based Low-Light Frame Interpolation with Per-Scene Optimization | SIGGRAPH Asia 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://openimaginglab.github.io/Sim2Real/) | [![GitHub](https://img.shields.io/github/stars/OpenImagingLab/sim2real)](https://github.com/OpenImagingLab/sim2real) |
|  |
|  |


## Event-based 3D Reconstruction

### Geometric & Semi-Dense Reconstruction

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EMVS` | [IJCV Paper](https://rpg.ifi.uzh.ch/docs/IJCV17_Rebecq.pdf)<br>EMVS: Event-based Multi-View Stereo—3D Reconstruction with an Event Camera in Real-Time | IJCV 2018 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_emvs)](https://github.com/uzh-rpg/rpg_emvs) |
| - | [![arXiv](https://img.shields.io/badge/arXiv-1807.07429-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1807.07429)<br>Semi-Dense 3D Reconstruction with a Stereo Event Camera | ECCV 2018 | - | - |
| `EventPS` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_EventPS_Real-Time_Photometric_Stereo_Using_an_Event_Camera_CVPR_2024_paper.pdf)<br>EventPS: Real-Time Photometric Stereo Using an Event Camera | CVPR 2024 | - | [Codeberg](https://codeberg.org/ybh1998/EventPS) |
|  |
|  |


### Neural Radiance Fields (NeRF) with Events

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `Ev-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2206.12455-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2206.12455)<br>Ev-NeRF: Event Based Neural Radiance Field | WACV 2023 | - | - |
| `EventNeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2206.11896-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2206.11896)<br>EventNeRF: Neural Radiance Fields from a Single Colour Event Camera | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/r00tman/EventNeRF)](https://github.com/r00tman/EventNeRF) |
| `E-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2208.11300-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2208.11300)<br>E-NeRF: Neural Radiance Fields from a Moving Event Camera | RA-L 2023 | - | [![GitHub](https://img.shields.io/github/stars/knelk/enerf)](https://github.com/knelk/enerf) |
| `E2NeRF` | [ICCV Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_E2NeRF_Event_Enhanced_Neural_Radiance_Fields_from_Blurry_Images_ICCV_2023_paper.html)<br>E2NeRF: Event Enhanced Neural Radiance Fields from Blurry Images | ICCV 2023 | - | [![GitHub](https://img.shields.io/github/stars/iCVTEAM/E2NeRF)](https://github.com/iCVTEAM/E2NeRF) |
| `Robust-e-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2309.08596-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2309.08596)<br>Robust e-NeRF: NeRF from Sparse & Noisy Events under Non-Uniform Motion | ICCV 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://wengflow.github.io/robust-e-nerf/) | [![GitHub](https://img.shields.io/github/stars/wengflow/robust-e-nerf)](https://github.com/wengflow/robust-e-nerf) |
| `DE-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2309.08416-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2309.08416)<br>Deformable Neural Radiance Fields using RGB and Event Cameras | ICCV 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://qimaqi.github.io/DE-NeRF.github.io/) | [![GitHub](https://img.shields.io/github/stars/qimaqi/DE-NeRF)](https://github.com/qimaqi/DE-NeRF) |
| `EvDNeRF` | [WACV Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Bhattacharya_Evdnerf_Reconstructing_Event_Data_With_Dynamic_Neural_Radiance_Fields_WACV_2024_paper.pdf)<br>EvDNeRF: Reconstructing Event Data with Dynamic Neural Radiance Fields | WACV 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.anishbhattacharya.com/research/evdnerf) | [![GitHub](https://img.shields.io/github/stars/anish-bhattacharya/EvDNeRF)](https://github.com/anish-bhattacharya/EvDNeRF) |
| `PAEv3d` | [![arXiv](https://img.shields.io/badge/arXiv-2401.17121-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2401.17121)<br>Physical Priors Augmented Event-Based 3D Reconstruction | ICRA 2024 | - | [![GitHub](https://img.shields.io/github/stars/Mercerai/PAEv3d)](https://github.com/Mercerai/PAEv3d) |
| `EvDeblurNeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2403.19780-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.19780)<br>Mitigating Motion Blur in Neural Radiance Fields with Events and Frames | CVPR 2024 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/EvDeblurNeRF)](https://github.com/uzh-rpg/EvDeblurNeRF) |
| `EBAD-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2406.14360-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.14360)<br>Deblurring Neural Radiance Fields with Event-Driven Bundle Adjustment | ACM MM 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://icvteam.github.io/EBAD-NeRF.html) | [![GitHub](https://img.shields.io/github/stars/iCVTEAM/EBAD-NeRF)](https://github.com/iCVTEAM/EBAD-NeRF) |
| `BeNeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2407.02174-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2407.02174)<br>BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream | ECCV 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://akawincent.github.io/BeNeRF/) | [![GitHub](https://img.shields.io/github/stars/WU-CVGL/BeNeRF)](https://github.com/WU-CVGL/BeNeRF) |
| `E3NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2408.01840-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2408.01840)<br>E3NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images | arXiv 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://icvteam.github.io/E3NeRF.html) | [![GitHub](https://img.shields.io/github/stars/iCVTEAM/E3NeRF)](https://github.com/iCVTEAM/E3NeRF) |
| `Deblur-e-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2409.17988-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.17988)<br>Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions | ECCV 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://wengflow.github.io/deblur-e-nerf/) | [![GitHub](https://img.shields.io/github/stars/wengflow/deblur-e-nerf)](https://github.com/wengflow/deblur-e-nerf) |
| `Event-ID` | [ACM MM Paper](https://dl.acm.org/doi/abs/10.1145/3664647.3681133)<br>Event-ID: Intrinsic Decomposition Using an Event Camera | ACM MM 2024 | - | - |
| `AE-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2501.02807-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2501.02807)<br>AE-NeRF: Augmenting Event-Based Neural Radiance Fields for Non-ideal Conditions and Larger Scenes | AAAI 2025 | - | - |
| `EvHDR-NeRF` | EvHDR-NeRF: Building High Dynamic Range Radiance Fields with Single Exposure Images and Events | AAAI 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://zehaoc.github.io/EvHDR-NeRF/) | [![GitHub](https://img.shields.io/github/stars/Zehaoc/EvHDR-NeRF)](https://github.com/Zehaoc/EvHDR-NeRF) |
| `LSE-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2409.06104-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.06104)<br>LSE-NeRF: Learning Sensor Modeling Errors for Deblurred Neural Radiance Fields with RGB-Event Stereo | 3DV 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://ubc-vision.github.io/LSENeRF/) | [![GitHub](https://img.shields.io/github/stars/ubc-vision/LSENeRF)](https://github.com/ubc-vision/LSENeRF) |
| `SaENeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2504.16389-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2504.16389)<br>SaENeRF: Suppressing Artifacts in Event-based Neural Radiance Fields | arXiv 2025 | - | [![GitHub](https://img.shields.io/github/stars/Mr-firework/SaENeRF)](https://github.com/Mr-firework/SaENeRF) |
| `DynEventNeRF` | Dynamic EventNeRF: Reconstructing General Dynamic Scenes from Multi-view RGB and Event Streams | CVPR Workshop 2025 | - | [![GitHub](https://img.shields.io/github/stars/r00tman/DynEventNeRF)](https://github.com/r00tman/DynEventNeRF) |
| `EventNeuS` | [![arXiv](https://img.shields.io/badge/arXiv-2602.03847-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2602.03847)<br>EventNeuS: 3D Mesh Reconstruction from a Single Event Camera | 3DV 2026 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://4dqv.mpi-inf.mpg.de/EventNeuS/) | - |
|  |
|  |


### 3D Gaussian Splatting (3DGS) with Events

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EvGGS` | [![arXiv](https://img.shields.io/badge/arXiv-2405.14959-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2405.14959)<br>EvGGS: A Collaborative Learning Framework for Event-based Generalizable Gaussian Splatting | arXiv 2024 | - | [![GitHub](https://img.shields.io/github/stars/Mercerai/EvGGS)](https://github.com/Mercerai/EvGGS) |
| `EvaGaussians` | [![arXiv](https://img.shields.io/badge/arXiv-2405.20224-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2405.20224)<br>EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images | ICCV 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://drexubery.github.io/EvaGaussians/) | [![GitHub](https://img.shields.io/github/stars/PKU-YuanGroup/EvaGaussians)](https://github.com/PKU-YuanGroup/EvaGaussians) |
| `Event3DGS` | [![arXiv](https://img.shields.io/badge/arXiv-2406.02972-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.02972)<br>Event3DGS: Event-Based 3D Gaussian Splatting for High-Speed Robot Egomotion | CoRL 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://tyxiong23.github.io/event3dgs) | - |
| `EaDeblur-GS` | [![arXiv](https://img.shields.io/badge/arXiv-2407.13520-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2407.13520)<br>EaDeblur-GS: Event Assisted 3D Deblur Reconstruction with Gaussian Splatting | arXiv 2024 | - | - |
| `Ev-GS` | [![arXiv](https://img.shields.io/badge/arXiv-2407.11343-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2407.11343)<br>Ev-GS: Event-Based Gaussian Splatting for Efficient and Accurate Radiance Field Rendering | MLSP 2024 | - | - |
| `E2GS` | [IEEE Paper](https://ieeexplore.ieee.org/document/10647607)<br>E2GS: Event Enhanced Gaussian Splatting | ICIP 2024 | - | [![GitHub](https://img.shields.io/github/stars/deguchihiroyuki/E2GS)](https://github.com/deguchihiroyuki/E2GS) |
| `Ev3DGS` | [IEEE Paper](https://ieeexplore.ieee.org/abstract/document/10848695)<br>Ev3DGS: Event Enhanced 3D Gaussian Splatting from Blurry Images | APSIPA ASC 2024 | - | - |
| `IncEventGS` | [![arXiv](https://img.shields.io/badge/arXiv-2410.08107-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2410.08107)<br>IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera | CVPR 2025 | - | [![GitHub](https://img.shields.io/github/stars/wu-cvgl/IncEventGS)](https://github.com/wu-cvgl/IncEventGS) |
| `EF-3DGS` | [![arXiv](https://img.shields.io/badge/arXiv-2410.15392-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2410.15392)<br>EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting | arXiv 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://lbh666.github.io/ef-3dgs/) | - |
| `E-3DGS` | [![arXiv](https://img.shields.io/badge/arXiv-2410.16995-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2410.16995)<br>E-3DGS: Gaussian Splatting with Exposure and Motion Events | arXiv 2024 | - | [![GitHub](https://img.shields.io/github/stars/MasterHow/E-3DGS)](https://github.com/MasterHow/E-3DGS) |
| `EventBoosted-3DGS` | [![arXiv](https://img.shields.io/badge/arXiv-2411.16180-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2411.16180)<br>Event-Boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction | ICCV 2025 | - | - |
| `Event-3DGS` | [NeurIPS Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e73ad1f690542144ce354637bb913c35-Abstract-Conference.html)<br>Event-3DGS: Event-Based 3D Reconstruction Using 3D Gaussian Splatting | NeurIPS 2024 | - | [![GitHub](https://img.shields.io/github/stars/lanpokn/Event-3DGS)](https://github.com/lanpokn/Event-3DGS) |
| `SweepEvGS` | [![arXiv](https://img.shields.io/badge/arXiv-2412.11579-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2412.11579)<br>SweepEvGS: Event-Based 3D Gaussian Splatting for Macro and Micro Radiance Field Rendering from a Single Sweep | TCSVT 2025 | - | - |
| `EventSplat` | [![arXiv](https://img.shields.io/badge/arXiv-2412.07293-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2412.07293)<br>EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-Time Rendering | CVPR 2025 | - | - |
| `BeSplat` | [WACV Paper](https://openaccess.thecvf.com/content/WACV2025W/EVGEN/papers/Matta_BeSplat_Gaussian_Splatting_from_a_Single_Blurry_Image_and_Event_WACVW_2025_paper.pdf)<br>BeSplat: Gaussian Splatting from a Single Blurry Image and Event Stream | WACVW 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://gopirajumatta.github.io/BeSplat/) | [![GitHub](https://img.shields.io/github/stars/GopiRajuMatta/BeSplat)](https://github.com/GopiRajuMatta/BeSplat) |
| `DiET-GS` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_DiET-GS_Diffusion_Prior_and_Event_Stream-Assisted_Motion_Deblurring_3D_Gaussian_CVPR_2025_paper.html)<br>DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting | CVPR 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://diet-gs.github.io/) | [![GitHub](https://img.shields.io/github/stars/DiET-GS/DiET-GS)](https://github.com/DiET-GS/DiET-GS) |
| `EBAD-Gaussian` | [![arXiv](https://img.shields.io/badge/arXiv-2504.10012-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2504.10012)<br>EBAD-Gaussian: Event-Driven Bundle Adjusted Deblur Gaussian Splatting | arXiv 2025 | - | - |
|  |
|  |


### Generalizable 3D Reconstruction

> :timer_clock: In chronological order, from the earliest to the latest.

<!-- TODO: This is an emerging direction beyond per-scene optimization. More feed-forward/generalizable methods and large-model-inspired 3D reconstruction works need to be surveyed and added here. -->

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `Elite-EvGS` | [![arXiv](https://img.shields.io/badge/arXiv-2409.13392-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.13392)<br>Elite-EvGS: Learning Event-based 3D Gaussian Splatting by Distilling Event-to-Video Priors | arXiv 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlislab22.github.io/elite-evgs/) | - |
|  |
|  |



# 4. Event Camera Understanding

### :one:
...

### :two: 
...

### :three: 
...




# 5. Applications

### :one:
...

### :two: 
...

### :three: 
...




# 6. Other Resources

### Tutorials

...



### Talks & Seminars

...



### Relevant Surveys

| Paper | Venue | Website | GitHub | 
|:-|:-:|:-:|:-:|
||
| [![arXiv](https://img.shields.io/badge/arXiv-1904.08405-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1904.08405)<br>Event-Based Vision: A Survey | TPAMI 2022 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/event-based_vision_resources)](https://github.com/uzh-rpg/event-based_vision_resources) |
| [![arXiv](https://img.shields.io/badge/arXiv-2302.08890-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2302.08890)<br>Deep Learning for Event-Based Vision: A Comprehensive Survey and Benchmarks | arXiv 2023 | - | [![GitHub](https://img.shields.io/github/stars/vlislab22/Deep-Learning-for-Event-based-Vision)](https://github.com/vlislab22/Deep-Learning-for-Event-based-Vision) |
| [![arXiv](https://img.shields.io/badge/arXiv-2304.09793-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2304.09793)<br>Event-Based Simultaneous Localization and Mapping: A Comprehensive Survey | arXiv 2023 | - | [![GitHub](https://img.shields.io/github/stars/kun150kun/ESLAM-survey)](https://github.com/kun150kun/ESLAM-survey) |
| [![arXiv](https://img.shields.io/badge/arXiv-24xx.xxxxx-b31b1b?style=flat-square&logo=arxiv)](https://www.mdpi.com/2078-2489/15/8/472)<br>An Application-Driven Survey on Event-Based Neuromorphic Computer Vision | Information 2024 | - | - |
| [![arXiv](https://img.shields.io/badge/arXiv-24xx.xxxxx-b31b1b?style=flat-square&logo=arxiv)](https://ieeexplore.ieee.org/abstract/document/10494342/)<br>Event Cameras in Automotive Sensing: A Review | IEEE Access 2024 | - | - |
| [![arXiv](https://img.shields.io/badge/arXiv-2408.13627-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2408.13627)<br>Recent Event Camera Innovations: A Survey | ECCVW 2024 | - | [![GitHub](https://img.shields.io/github/stars/chakravarthi589/Event-based-Vision_Resources)](https://github.com/chakravarthi589/Event-based-Vision_Resources) |
| [![arXiv](https://img.shields.io/badge/arXiv-2405.03995-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2405.03995)<br>Deep Event-Based Object Detection in Autonomous Driving: A Survey | arXiv 2024 | - | - |
| [![arXiv](https://img.shields.io/badge/arXiv-2409.17680-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.17680)<br>Event-Based Stereo Depth Estimation: A Survey | TPAMI 2025 | - | [![GitHub](https://img.shields.io/github/stars/tub-rip/EventStereoSurvey)](https://github.com/tub-rip/EventStereoSurvey) |
| [![arXiv](https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b?style=flat-square&logo=arxiv)](https://www.mdpi.com/1424-8220/26/1/81)<br>Event-Based Vision Application on Autonomous Unmanned Aerial Vehicle: A Systematic Review of Prospects and Challenges | Sensors 2025 | - | - |
| [![arXiv](https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b?style=flat-square&logo=arxiv)](https://dl.acm.org/doi/abs/10.1145/3786332)<br>Event Camera Meets Mobile Embodied Perception: Abstraction, Algorithm, Acceleration, Application | ACM Computing Surveys 2025 | - | - |
| [![arXiv](https://img.shields.io/badge/arXiv-2505.08438-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2505.08438)<br>A Survey of 3D Reconstruction with Event Cameras | arXiv 2025 | - | - |
| [![arXiv](https://img.shields.io/badge/arXiv-2509.09971-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2509.09971)<br>Event Camera Guided Visual Media Restoration & 3D Reconstruction: A Survey | arXiv 2025 | - | - |
|  |



# 7. Acknowledgements

To be added.
