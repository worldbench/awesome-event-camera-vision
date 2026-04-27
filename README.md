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
- [**2. Event Camera Perception**](#2-event-camera-perception)
    - [2.1 Object Detection](#21-object-detection)
        - [CNN, RNN \& Graph Approaches](#cnn-rnn--graph-approaches)
        - [Transformer \& Attention-Based](#transformer--attention-based)
        - [SNN \& Efficient Detection](#snn--efficient-detection)
    - [2.2 Semantic Segmentation](#22-semantic-segmentation)
        - [Supervised \& Multi-Modal Fusion](#supervised--multi-modal-fusion)
        - [Annotation-Efficient \& Adaptation](#annotation-efficient--adaptation)
        - [Motion \& Instance Segmentation](#motion--instance-segmentation)
    - [2.3 Depth \& Optical Flow Estimation](#23-depth--optical-flow-estimation)
        - [Monocular Depth Estimation](#monocular-depth-estimation)
        - [Stereo Depth Estimation](#stereo-depth-estimation)
        - [Optical Flow Estimation](#optical-flow-estimation)
    - [2.4 Object Tracking](#24-object-tracking)
        - [Frame-Event Tracking](#frame-event-tracking)
        - [Event-Only Tracking](#event-only-tracking)
        - [Long-Term \& Cross-Modal Tracking](#long-term--cross-modal-tracking)
    - [2.5 Action Recognition \& Pose Estimation](#25-action-recognition--pose-estimation)
        - [Action \& Gesture Recognition](#action--gesture-recognition)
        - [Body Pose Estimation](#body-pose-estimation)
        - [Hand \& Egocentric Pose Estimation](#hand--egocentric-pose-estimation)
- [**3. Event Camera Reconstruction**](#3-event-camera-reconstruction)
    - [3.1 Event-to-Video Reconstruction](#31-event-to-video-reconstruction)
        - [Discriminative Models](#discriminative-models)
        - [Generative Models](#generative-models)
        - [Self-Supervised Approaches](#self-supervised-approaches)
    - [3.2 Video Frame Interpolation \& Deblurring](#32-video-frame-interpolation--deblurring)
        - [Frame Interpolation](#frame-interpolation)
        - [Motion Deblurring](#motion-deblurring)
        - [Joint Interpolation \& Deblurring](#joint-interpolation--deblurring)
    - [3.3 Image Enhancement](#33-image-enhancement)
        - [Super-Resolution](#super-resolution)
        - [HDR Reconstruction](#hdr-reconstruction)
        - [Low-Light Enhancement](#low-light-enhancement)
    - [3.4 3D Reconstruction](#34-3d-reconstruction)
        - [Geometric \& Semi-Dense](#geometric--semi-dense)
        - [NeRF-Based](#nerf-based)
        - [3DGS-Based](#3dgs-based)
- [**4. Event Camera Understanding**](#4-event-camera-understanding)
    - [4.1 Open-Vocabulary Perception](#41-open-vocabulary-perception)
        - [Image-Level Recognition](#image-level-recognition)
        - [Dense Segmentation \& Detection](#dense-segmentation--detection)
        - [Self-Supervised Pre-training](#self-supervised-pre-training)
        - [Cross-Modal Transfer](#cross-modal-transfer)
    - [4.2 Scene Understanding \& Reasoning](#42-scene-understanding--reasoning)
        - [Event-Based MLLMs](#event-based-mllms)
        - [Visual Grounding \& Embodied Intelligence](#visual-grounding--embodied-intelligence)
        - [Language-Guided Recognition](#language-guided-recognition)
    - [4.3 Event Data Simulation \& Generation](#43-event-data-simulation--generation)
        - [Physics-Based Simulators](#physics-based-simulators)
        - [Learned \& Neural Generation](#learned--neural-generation)
        - [Sim-to-Real Transfer](#sim-to-real-transfer)
- [**5. Datasets \& Benchmarks**](#5-datasets--benchmarks)
    - [Benchmarks](#benchmarks)
    - [Workshops](#workshops)
    - [Datasets](#datasets)
    - [Simulators](#simulators)
- [**6. Challenges \& Future Directions**](#6-challenges--future-directions)
- [**7. Other Resources**](#7-other-resources)
    - [Tutorials](#tutorials)
    - [Talks \& Seminars](#talks--seminars)
    - [Relevant Surveys](#relevant-surveys)
- [**8. Acknowledgements**](#8-acknowledgements)



# Background

...




# 2. Event Camera Perception

## 2.1 Object Detection

### CNN, RNN & Graph Approaches

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `DVS-Detection` | [![arXiv](https://img.shields.io/badge/arXiv-1709.09323-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1709.09323)<br>Pseudo-Labels for Supervised Learning on Dynamic Vision Sensor Data, Applied to Object Detection under Ego-Motion | CVPRW 2018 | - | - |
| `YOLE` | [![arXiv](https://img.shields.io/badge/arXiv-1805.07931-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1805.07931)<br>Asynchronous Convolutional Networks for Object Detection in Neuromorphic Cameras | CVPRW 2019 | - | - |
| `MatrixLSTM` | [![arXiv](https://img.shields.io/badge/arXiv-2009.13436-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2009.13436)<br>Learning to Detect Objects with a 1 Megapixel Event Camera | NeurIPS 2020 | - | - |
| `ASTMNet` | [Asynchronous Spatio-Temporal Memory Network for Continuous Event-Based Object Detection](https://ieeexplore.ieee.org/document/9749022)<br>Asynchronous Spatio-Temporal Memory Network for Continuous Event-Based Object Detection | TIP 2022 | - | - |
| `AEGNN` | [![arXiv](https://img.shields.io/badge/arXiv-2203.17149-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2203.17149)<br>AEGNN: Asynchronous Event-based Graph Neural Networks | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/aegnn)](https://github.com/uzh-rpg/aegnn) |
| `RENet` | [![arXiv](https://img.shields.io/badge/arXiv-2209.08323-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2209.08323)<br>RGB-Event Fusion for Moving Object Detection in Autonomous Driving | ICRA 2023 | - | [![GitHub](https://img.shields.io/github/stars/ZZY-Zhou/RENet)](https://github.com/ZZY-Zhou/RENet) |
| `DAGR` | Low-Latency Automotive Vision with Event Cameras | Nature 2024 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/dagr)](https://github.com/uzh-rpg/dagr) |
|  |
|  |


### Transformer & Attention-Based

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `RVT` | [![arXiv](https://img.shields.io/badge/arXiv-2212.05598-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2212.05598)<br>Recurrent Vision Transformers for Object Detection with Event Cameras | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/RVT)](https://github.com/uzh-rpg/RVT) |
| `SODFormer` | [![arXiv](https://img.shields.io/badge/arXiv-2308.04047-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2308.04047)<br>SODFormer: Streaming Object Detection with Transformer Using Events and Frames | TPAMI 2023 | - | - |
| `GET` | [![arXiv](https://img.shields.io/badge/arXiv-2310.02642-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2310.02642)<br>GET: Group Event Transformer for Event-Based Vision | ICCV 2023 | - | [![GitHub](https://img.shields.io/github/stars/Peterande/GET-Group-Event-Transformer)](https://github.com/Peterande/GET-Group-Event-Transformer) |
| `LEOD` | [![arXiv](https://img.shields.io/badge/arXiv-2311.17286-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2311.17286)<br>LEOD: Label-Efficient Object Detection for Event Cameras | CVPR 2024 | - | [![GitHub](https://img.shields.io/github/stars/Wuziyi616/LEOD)](https://github.com/Wuziyi616/LEOD) |
| `State-Space` | [![arXiv](https://img.shields.io/badge/arXiv-2402.15584-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2402.15584)<br>State Space Models for Event Cameras | CVPR 2024 (Spotlight) | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/ssms_event_cameras)](https://github.com/uzh-rpg/ssms_event_cameras) |
| `SAST` | [![arXiv](https://img.shields.io/badge/arXiv-2404.01882-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2404.01882)<br>Scene Adaptive Sparse Transformer for Event-based Object Detection | CVPR 2024 | - | [![GitHub](https://img.shields.io/github/stars/Peterande/SAST)](https://github.com/Peterande/SAST) |
|  |
|  |


### SNN & Efficient Detection

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `SFOD` | [![arXiv](https://img.shields.io/badge/arXiv-2403.15192-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.15192)<br>SFOD: Spiking Fusion Object Detector | CVPR 2024 | - | - |
| `MoE-HeatDet` | [![arXiv](https://img.shields.io/badge/arXiv-2412.06647-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2412.06647)<br>Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset | CVPR 2025 | - | - |
| `EventFly` | EventFly: Event Camera Perception from Ground to the Sky | CVPR 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://event-fly.github.io/) | - |
| `OpenESS-Det` | [![arXiv](https://img.shields.io/badge/arXiv-2510.00681-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2510.00681)<br>Adaptive Event Stream Slicing for Open-Vocabulary Event-Based Object Detection | arXiv 2025 | - | - |
|  |
|  |


## 2.2 Semantic Segmentation

### Supervised & Multi-Modal Fusion

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EV-SegNet` | [![arXiv](https://img.shields.io/badge/arXiv-1811.12039-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1811.12039)<br>EV-SegNet: Semantic Segmentation for Event-based Cameras | CVPRW 2019 | - | [![GitHub](https://img.shields.io/github/stars/Shathe/Ev-SegNet)](https://github.com/Shathe/Ev-SegNet) |
| `EvDistill` | [![arXiv](https://img.shields.io/badge/arXiv-2111.12341-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2111.12341)<br>EvDistill: Asynchronous Events to End-task Learning via Bidirectional Reconstruction-guided Cross-modal Knowledge Distillation | CVPR 2021 | - | [![GitHub](https://img.shields.io/github/stars/addisonwang2013/evdistill)](https://github.com/addisonwang2013/evdistill) |
| `ESS` | [![arXiv](https://img.shields.io/badge/arXiv-2203.10016-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2203.10016)<br>ESS: Learning Event-based Semantic Segmentation from Still Images | ECCV 2022 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/ess)](https://github.com/uzh-rpg/ess) |
| `EV-Transfer` | [![arXiv](https://img.shields.io/badge/arXiv-2109.02618-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2109.02618)<br>Bridging the Gap between Events and Frames through Unsupervised Domain Adaptation | RA-L 2022 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_ev-transfer)](https://github.com/uzh-rpg/rpg_ev-transfer) |
| `CMX` | [![arXiv](https://img.shields.io/badge/arXiv-2203.04838-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2203.04838)<br>CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation | TITS 2023 | - | [![GitHub](https://img.shields.io/github/stars/huaaaliu/RGBX_Semantic_Segmentation)](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) |
| `HMNet` | [![arXiv](https://img.shields.io/badge/arXiv-2305.17852-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2305.17852)<br>Hierarchical Neural Memory Network for Low Latency Event Processing | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/hamarh/HMNet_pth)](https://github.com/hamarh/HMNet_pth) |
| `HALSIE` | [![arXiv](https://img.shields.io/badge/arXiv-2211.10754-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2211.10754)<br>HALSIE: Hybrid Approach to Learning Segmentation by Simultaneously Exploiting Image and Event Modalities | WACV 2024 | - | - |
| `EventSAM` | [![arXiv](https://img.shields.io/badge/arXiv-2312.16222-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2312.16222)<br>Segment Any Event Streams via Weighted Adaptation of Pivotal Tokens | CVPR 2024 | - | [![GitHub](https://img.shields.io/github/stars/zhiwen-xdu/EventSAM)](https://github.com/zhiwen-xdu/EventSAM) |
| `SAM-E-Adapter` | SAM-Event-Adapter: Adapting Segment Anything Model for Event-RGB Semantic Segmentation | ICRA 2024 | - | - |
| `OpenESS` | [![arXiv](https://img.shields.io/badge/arXiv-2405.05259-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2405.05259)<br>OpenESS: Event-Based Semantic Scene Understanding with Open Vocabularies | CVPR 2024 (Highlight) | - | [![GitHub](https://img.shields.io/github/stars/ldkong1205/OpenESS)](https://github.com/ldkong1205/OpenESS) |
| `EvSAM` | EvSAM: Segment Anything Model with Event-based Assistance | ACM TOMM 2025 | - | - |
| `OVOSE` | [![arXiv](https://img.shields.io/badge/arXiv-2408.09424-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2408.09424)<br>OVOSE: Open-Vocabulary Semantic Segmentation in Event-Based Cameras | arXiv 2024 | - | - |
| `SEAL` | [![arXiv](https://img.shields.io/badge/arXiv-2601.23159-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2601.23159)<br>Segment Any Events with Language | ICLR 2026 | - | - |
|  |
|  |


### Annotation-Efficient & Adaptation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `HPL-ESS` | HPL-ESS: Hybrid Pseudo-Labeling for Unsupervised Event-Based Semantic Segmentation | CVPR 2024 | - | - |
| `EV-WSSS` | [![arXiv](https://img.shields.io/badge/arXiv-2407.11216-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2407.11216)<br>Finding Meaning in Points: Weakly Supervised Semantic Segmentation for Event Cameras | ECCV 2024 | - | - |
|  |
|  |


### Motion & Instance Segmentation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EV-IMO` | EV-IMO: Motion Segmentation Dataset and Learning Pipeline for Event Cameras | IROS 2019 | - | - |
| `EMSGC` | [![arXiv](https://img.shields.io/badge/arXiv-2012.08730-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2012.08730)<br>Event-Based Motion Segmentation with Spatio-Temporal Graph Cuts | TNNLS 2021 | - | - |
| `Un-EvMoSeg` | Unsupervised Event-Based Independent Motion Segmentation | ECCV 2024 | - | - |
|  |
|  |


## 2.3 Depth & Optical Flow Estimation

### Monocular Depth Estimation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EMDE` | [![arXiv](https://img.shields.io/badge/arXiv-2203.12270-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2203.12270)<br>Event-Based Dense Reconstruction Pipeline | ICRAS 2022 | - | - |
| `E2Depth` | [![arXiv](https://img.shields.io/badge/arXiv-2010.08350-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2010.08350)<br>Learning Monocular Dense Depth from Events | 3DV 2020 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_e2depth)](https://github.com/uzh-rpg/rpg_e2depth) |
| `RAM-Net` | [![arXiv](https://img.shields.io/badge/arXiv-2102.09320-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2102.09320)<br>Combining Events and Frames using Recurrent Asynchronous Multimodal Networks for Monocular Depth Prediction | RA-L 2021 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_ramnet)](https://github.com/uzh-rpg/rpg_ramnet) |
| `HMNet-Depth` | [![arXiv](https://img.shields.io/badge/arXiv-2305.17852-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2305.17852)<br>Hierarchical Neural Memory Network for Low Latency Event Processing | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/hamarh/HMNet_pth)](https://github.com/hamarh/HMNet_pth) |
| `EReFormer` | [![arXiv](https://img.shields.io/badge/arXiv-2212.02791-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2212.02791)<br>Event-based Monocular Depth Estimation with Recurrent Transformers | ECCV 2024 | - | [![GitHub](https://img.shields.io/github/stars/liuxu0303/EReFormer)](https://github.com/liuxu0303/EReFormer) |
| `EMoDE` | [![arXiv](https://img.shields.io/badge/arXiv-2412.19067-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2412.19067)<br>Learning Monocular Depth from Events via Egomotion Compensation | arXiv 2024 | - | - |
| `Depth AnyEvent` | Depth Any Event Stream: Enhancing Event-based Monocular Depth Estimation via Dense-to-Sparse Distillation | ICCV 2025 | - | - |
|  |
|  |

### Stereo Depth Estimation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `MVSEC-Stereo` | [![arXiv](https://img.shields.io/badge/arXiv-1801.10202-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1801.10202)<br>The Multi Vehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception | RA-L 2018 | - | - |
| `StereoSpike` | [![arXiv](https://img.shields.io/badge/arXiv-2109.13751-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2109.13751)<br>StereoSpike: Depth Learning with a Spiking Neural Network | arXiv 2021 | - | - |
| `DSGN-Event` | Discrete Time Convolution for Fast Event-based Stereo | CVPR 2022 | - | - |
| `SDE-Net` | Selection and Cross Similarity for Event-Image Deep Stereo | ECCV 2022 | - | [![GitHub](https://img.shields.io/github/stars/Chohoonhee/SCSNet)](https://github.com/Chohoonhee/SCSNet) |
| `EvAC3D` | [![arXiv](https://img.shields.io/badge/arXiv-2309.09513-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2309.09513)<br>Learning Parallax for Stereo Event-Based Motion Deblurring | ICCV 2023 | - | - |
| `E-Stereo-FUDA` | [![arXiv](https://img.shields.io/badge/arXiv-2210.08927-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2210.08927)<br>Event-Based Stereo Depth Estimation from Ego-Motion using Ray Density Fusion | ECCV 2024 | - | - |
| `ESVO2` | [![arXiv](https://img.shields.io/badge/arXiv-2410.09374-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2410.09374)<br>ESVO2: Direct Visual-Inertial Odometry with Stereo Event Cameras | TRO 2025 | - | - |
|  |
|  |


### Optical Flow Estimation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EV-FlowNet` | [![arXiv](https://img.shields.io/badge/arXiv-1802.06898-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1802.06898)<br>EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras | RSS 2018 | - | - |
| `Spike-FlowNet` | [![arXiv](https://img.shields.io/badge/arXiv-2003.06696-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2003.06696)<br>Spike-FlowNet: Event-Driven Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks | ECCV 2020 | - | [![GitHub](https://img.shields.io/github/stars/chan8972/Spike-FlowNet)](https://github.com/chan8972/Spike-FlowNet) |
| `E-RAFT` | [![arXiv](https://img.shields.io/badge/arXiv-2108.10552-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2108.10552)<br>E-RAFT: Dense Optical Flow from Event Cameras | 3DV 2022 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/E-RAFT)](https://github.com/uzh-rpg/E-RAFT) |
| `DCEIFlow` | [![arXiv](https://img.shields.io/badge/arXiv-2211.09078-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2211.09078)<br>Learning Dense and Continuous Optical Flow from an Event Camera | TIP 2022 | - | [![GitHub](https://img.shields.io/github/stars/danqu130/DCEIFlow)](https://github.com/danqu130/DCEIFlow) |
| `BlinkFlow` | [![arXiv](https://img.shields.io/badge/arXiv-2303.07716-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2303.07716)<br>BlinkFlow: A Dataset to Push the Limits of Event-based Optical Flow Estimation | IROS 2023 | - | - |
| `EGMBA` | [![arXiv](https://img.shields.io/badge/arXiv-2503.03307-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2503.03307)<br>Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers | CVPR 2025 | - | [![GitHub](https://img.shields.io/github/stars/jizhaox/relpose-event)](https://github.com/jizhaox/relpose-event) |
|  |
|  |


## 2.4 Object Tracking

### Frame-Event Tracking

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `FE108-Track` | [![arXiv](https://img.shields.io/badge/arXiv-2109.09052-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2109.09052)<br>Object Tracking by Jointly Exploiting Frame and Event Domain | ICCV 2021 | - | - |
| `CEUTrack` | [![arXiv](https://img.shields.io/badge/arXiv-2211.11010-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2211.11010)<br>Revisiting Color-Event based Tracking: A Unified Network, Dataset, and Metric | PR 2025 | - | - |
| `AFNet` | [![arXiv](https://img.shields.io/badge/arXiv-2305.15688-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2305.15688)<br>Frame-Event Alignment and Fusion Network for High Frame Rate Tracking | CVPR 2024 | - | - |
| `EventVOT-Track` | [![arXiv](https://img.shields.io/badge/arXiv-2309.14611-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2309.14611)<br>Event Stream-based Visual Object Tracking: A High-Resolution Benchmark Dataset and A Novel Baseline | CVPR 2024 | - | - |
|  |
|  |


### Event-Only Tracking

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `STNet` | Spiking Transformers for Event-Based Single Object Tracking | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/Jee-King/CVPR2022_STNet)](https://github.com/Jee-King/CVPR2022_STNet) |
| `HDETrack` | [![arXiv](https://img.shields.io/badge/arXiv-2401.02826-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2401.02826)<br>Cross-Resolution Object Tracking Using Unaligned Frame and Event Cameras | TMM 2025 | - | - |
| `eMoE-Tracker` | [![arXiv](https://img.shields.io/badge/arXiv-2406.20024-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.20024)<br>Environmental MoE-based Transformer for Robust Event-Guided Object Tracking | RA-L 2024 | - | - |
| `MambaEVT` | [![arXiv](https://img.shields.io/badge/arXiv-2404.18174-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2404.18174)<br>Mamba-FETrack: Frame-Event Tracking via State Space Model | arXiv 2024 | - | - |
|  |
|  |


### Long-Term & Cross-Modal Tracking

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `TETrack` | [![arXiv](https://img.shields.io/badge/arXiv-2403.05839-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.05839)<br>Long-Term Visual Object Tracking with Event Cameras | arXiv 2024 | - | - |
| `SiamEFT` | SiamEFT: Adaptive-Time Feature Extraction Hybrid Network for RGBE Multi-Domain Object Tracking | Front. Neurosci. 2024 | - | - |
| `EventFly-Track` | EventFly: Event Camera Perception from Ground to the Sky | CVPR 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://event-fly.github.io/) | - |
|  |
|  |


## 2.5 Action Recognition & Pose Estimation

### Action & Gesture Recognition

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `DVS-Gesture` | A Low Power, Fully Event-Based Gesture Recognition System | CVPR 2017 | - | - |
| `PointNet-EVS` | [![arXiv](https://img.shields.io/badge/arXiv-1904.07518-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1904.07518)<br>Space-Time Event Clouds for Gesture Recognition: From RGB Cameras to Event Cameras | CVPRW 2019 | - | - |
| `E2(GO)MOTION` | [![arXiv](https://img.shields.io/badge/arXiv-2112.03596-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2112.03596)<br>E2(GO)MOTION: Motion Augmented Event Stream for Egocentric Action Recognition | CVPR 2022 | - | - |
| `ExACT` | [![arXiv](https://img.shields.io/badge/arXiv-2403.12534-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.12534)<br>ExACT: Language-guided Conceptual Reasoning and Uncertainty Estimation for Event-based Action Recognition | CVPR 2024 | - | - |
| `HARDVS-Method` | [![arXiv](https://img.shields.io/badge/arXiv-2312.08849-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2312.08849)<br>HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors | AAAI 2024 | - | - |
| `DailyDVS-Method` | DailyDVS-200: A Comprehensive Benchmark Dataset for Event-Based Action Recognition | ECCV 2024 | - | - |
| `EventFly-Action` | EventFly: Event Camera Perception from Ground to the Sky | CVPR 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://event-fly.github.io/) | - |
|  |
|  |


### Body Pose Estimation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `DHP19-Method` | DHP19: Dynamic Vision Sensor 3D Human Pose Dataset | CVPRW 2019 | - | - |
| `EventCap` | [![arXiv](https://img.shields.io/badge/arXiv-1908.11505-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1908.11505)<br>EventCap: Monocular 3D Capture of High-Speed Human Motions using an Event Camera | CVPR 2020 | - | - |
| `EventHPE` | Lifting Monocular Events to 3D Human Poses | CVPR 2021 | - | - |
| `EventEgo3D` | [![arXiv](https://img.shields.io/badge/arXiv-2404.08640-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2404.08640)<br>EventEgo3D: 3D Human Motion Capture from Egocentric Event Streams | CVPR 2024 | - | - |
| `E-POSE` | E-POSE: A Large Scale Event Camera Dataset for Object Pose Estimation | Sci. Data 2025 | - | - |
| `EHPT-XC` | A Benchmark Dataset for Event-Guided Human Pose Estimation and Tracking in Extreme Conditions | NeurIPS 2024 | - | - |
|  |
|  |


### Hand & Egocentric Pose Estimation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EvHandPose` | [![arXiv](https://img.shields.io/badge/arXiv-2303.02862-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2303.02862)<br>EvHandPose: Event-based 3D Hand Pose Estimation with Sparse Supervision | TPAMI 2024 | - | - |
| `EventHandPose` | Complementing Event Streams and RGB Frames for Hand Mesh Reconstruction | ICCV 2023 | - | - |
| `EvRGBHand` | EvRGBHand: Complementing Event Streams and RGB Frames for Hand Mesh Reconstruction | CVPR 2024 | - | - |
|  |
|  |



# 3. Event Camera Reconstruction

> See [Section 5: Datasets & Benchmarks](#5-datasets--benchmarks) for reconstruction-specific datasets.


## 3.1 Event-to-Video Reconstruction

### Discriminative Models

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `E2VID` | [![arXiv](https://img.shields.io/badge/arXiv-1904.08298-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1904.08298)<br>Events-to-Video: Bringing Modern Computer Vision to Event Cameras | CVPR 2019 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_e2vid)](https://github.com/uzh-rpg/rpg_e2vid) |
| `E2VID+` | [![arXiv](https://img.shields.io/badge/arXiv-2003.09078-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2003.09078)<br>Reducing the Sim-to-Real Gap for Event Cameras | ECCV 2020 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://timostoff.github.io/20ecnn) | [![GitHub](https://img.shields.io/github/stars/TimoStoff/event_cnn_minimal)](https://github.com/TimoStoff/event_cnn_minimal) |
| `SPADE-E2VID` | [IEEE Paper](https://ieeexplore.ieee.org/abstract/document/9337171)<br>SPADE-E2VID: Spatially-Adaptive Denormalization for Event-Based Video Reconstruction | TIP 2021 | - | [![GitHub](https://img.shields.io/github/stars/RodrigoGantier/SPADE_E2VID)](https://github.com/RodrigoGantier/SPADE_E2VID) |
| `ET-Net` | [ICCV Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Weng_Event-Based_Video_Reconstruction_Using_Transformer_ICCV_2021_paper.pdf)<br>Event-based Video Reconstruction Using Transformer | ICCV 2021 | - | [![GitHub](https://img.shields.io/github/stars/warranweng/et-net)](https://github.com/warranweng/et-net) |
| `EVSNN` | [![arXiv](https://img.shields.io/badge/arXiv-2201.10943-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2201.10943)<br>Event-based Video Reconstruction via Potential-assisted Spiking Neural Network | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/LinZhu111/EVSNN)](https://github.com/LinZhu111/EVSNN) |
| `E-SAI` | [TPAMI Paper](https://ieeexplore.ieee.org/document/9740355)<br>Learning to See Through with Events | TPAMI 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://dvs-whu.cn/projects/esai/) | [![GitHub](https://img.shields.io/github/stars/dvs-whu/E-SAI)](https://github.com/dvs-whu/E-SAI) |
| `HyperE2VID` | [![arXiv](https://img.shields.io/badge/arXiv-2305.06382-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2305.06382)<br>HyperE2VID: Improving Event-Based Video Reconstruction via Hypernetworks | TIP 2024 | - | [![GitHub](https://img.shields.io/github/stars/ercanburak/HyperE2VID)](https://github.com/ercanburak/HyperE2VID) |
| `STLR` | [ECCV Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05843.pdf)<br>Spike-temporal Latent Representation for Energy-Efficient Event-to-Video Reconstruction | ECCV 2024 | - | - |
| `MamEVSR` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_Event-based_Video_Super-Resolution_via_State_Space_Models_CVPR_2025_paper.pdf)<br>Event-based Video Super-Resolution via State Space Models | CVPR 2025 | - | - |
|  |
|  |


### Generative Models

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `cGAN-E2V` | [CVPR Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Event-Based_High_Dynamic_Range_Image_and_Very_High_Frame_Rate_CVPR_2019_paper.pdf)<br>Event-Based High Dynamic Range Image and Very High Frame Rate Video Generation Using Conditional Generative Adversarial Networks | CVPR 2019 | - | - |
| `E2VIDiff` | [![arXiv](https://img.shields.io/badge/arXiv-2407.08231-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2407.08231)<br>E2VIDiff: Perceptual Events-to-Video Reconstruction using Diffusion Priors | arXiv 2024 | - | - |
| `TRG-Diffusion` | [ECCV Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05690.pdf)<br>Temporal Residual Guided Diffusion Framework for Event-Driven Video Reconstruction | ECCV 2024 | - | - |
|  |
|  |


### Self-Supervised Approaches

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| - | [ECCV Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630647.pdf)<br>Learning to See in the Dark with Events | ECCV 2020 | - | - |
| - | [![arXiv](https://img.shields.io/badge/arXiv-2009.08283-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2009.08283)<br>Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy | CVPR 2021 | - | - |
| `EvDeraining` | [ICCV Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Unsupervised_Video_Deraining_with_An_Event_Camera_ICCV_2023_paper.pdf)<br>Unsupervised Video Deraining with an Event Camera | ICCV 2023 | - | [![GitHub](https://img.shields.io/github/stars/booker-max/Unsupervised-Deraining-with-Event-Camera)](https://github.com/booker-max/Unsupervised-Deraining-with-Event-Camera) |
| `EvINR` | [![arXiv](https://img.shields.io/badge/arXiv-2407.18500-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2407.18500)<br>Revisit Event Generation Model: Self-Supervised Learning of Event-to-Video Reconstruction with Implicit Neural Representations | ECCV 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlislab22.github.io/EvINR/) | [![GitHub](https://img.shields.io/github/stars/wzpscott/EvINR)](https://github.com/wzpscott/EvINR) |
|  |
|  |


## 3.2 Video Frame Interpolation & Deblurring

### Frame Interpolation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `TimeLens` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Tulyakov_Time_Lens_Event-Based_Video_Frame_Interpolation_CVPR_2021_paper.html)<br>TimeLens: Event-based Video Frame Interpolation | CVPR 2021 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://rpg.ifi.uzh.ch/TimeLens.html) | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_timelens)](https://github.com/uzh-rpg/rpg_timelens) |
| `TimeReplayer` | [![arXiv](https://img.shields.io/badge/arXiv-2203.13859-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2203.13859)<br>TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation | CVPR 2022 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://sites.google.com/view/timereplayer) | - |
| `CBMNet` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Event-Based_Video_Frame_Interpolation_With_Cross-Modal_Asymmetric_Bidirectional_Motion_Fields_CVPR_2023_paper.html)<br>Event-Based Video Frame Interpolation With Cross-Modal Asymmetric Bidirectional Motion Fields | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/intelpro/CBMNet)](https://github.com/intelpro/CBMNet) |
| `REFID` | [![arXiv](https://img.shields.io/badge/arXiv-2301.05191-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2301.05191)<br>Event-Based Frame Interpolation with Ad-hoc Deblurring | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/AHupuJR/REFID)](https://github.com/AHupuJR/REFID) |
| `Revisit-EBVFI` | [![arXiv](https://img.shields.io/badge/arXiv-2307.12558-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2307.12558)<br>Revisiting Event-Based Video Frame Interpolation | IROS 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://jiabenchen.github.io/revisit_event) | - |
| `SuperFast` | [IEEE Paper](https://ieeexplore.ieee.org/document/9962797)<br>SuperFast: 200x Video Frame Interpolation via Event Camera | TPAMI 2023 | - | [![GitHub](https://img.shields.io/github/stars/lisiqi19971013/SuperFast)](https://github.com/lisiqi19971013/SuperFast) |
| `EVFI-DS` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Video_Frame_Interpolation_via_Direct_Synthesis_with_the_Event-based_Reference_CVPR_2024_paper.html)<br>Video Frame Interpolation via Direct Synthesis with the Event-based Reference | CVPR 2024 | - | - |
| `TimeLens-XL` | [ECCV Paper](https://link.springer.com/chapter/10.1007/978-3-031-73226-3_11)<br>TimeLens-XL: Real-Time Event-Based Video Frame Interpolation with Large Motion | ECCV 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://openimaginglab.github.io/TimeLens-XL/) | [![GitHub](https://img.shields.io/github/stars/OpenImagingLab/TimeLens-XL)](https://github.com/OpenImagingLab/TimeLens-XL) |
| `REVDM` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Repurposing_Pre-trained_Video_Diffusion_Models_for_Event-based_Video_Interpolation_CVPR_2025_paper.html)<br>Repurposing Pre-trained Video Diffusion Models for Event-based Video Interpolation | CVPR 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vdm-evfi.github.io/) | [![GitHub](https://img.shields.io/github/stars/codingrex/VDM_EVFI)](https://github.com/codingrex/VDM_EVFI) |
|  |
|  |


### Motion Deblurring

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `mEDI` | [IEEE Paper](https://ieeexplore.ieee.org/abstract/document/9252186)<br>Bringing a Blurry Frame Alive at High Frame-Rate with an Event Camera | CVPR 2019 | - | [![GitHub](https://img.shields.io/github/stars/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera)](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera) |
| `EFNet` | [ECCV Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780403.pdf)<br>Event-Based Fusion for Motion Deblurring with Cross-modal Attention | ECCV 2022 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://ahupujr.github.io/EFNet/) | [![GitHub](https://img.shields.io/github/stars/AHupuJR/EFNet)](https://github.com/AHupuJR/EFNet) |
| `SAN` | [![arXiv](https://img.shields.io/badge/arXiv-2308.05932-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2308.05932)<br>Generalizing Event-Based Motion Deblurring in Real-World Scenarios | ICCV 2023 | - | [![GitHub](https://img.shields.io/github/stars/xiangz-0/gem)](https://github.com/xiangz-0/gem) |
| `DA-Deblur` | [ECCV Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06299.pdf)<br>Motion Aware Event Representation-Driven Image Deblurring | ECCV 2024 | - | [![GitHub](https://img.shields.io/github/stars/ZhijingS/DA_event_deblur)](https://github.com/ZhijingS/DA_event_deblur) |
| `ELEDNet` | [ECCV Paper](https://link.springer.com/chapter/10.1007/978-3-031-73254-6_25)<br>Towards Real-World Event-Guided Low-Light Video Enhancement and Deblurring | ECCV 2024 | - | [![GitHub](https://img.shields.io/github/stars/intelpro/ELEDNet)](https://github.com/intelpro/ELEDNet) |
| `EGDeblurring` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Diffusion-based_Event_Generation_for_High-Quality_Image_Deblurring_CVPR_2025_paper.html)<br>Diffusion-based Event Generation for High-Quality Image Deblurring | CVPR 2025 | - | [![GitHub](https://img.shields.io/github/stars/XinanXie/EGDeblurring)](https://github.com/XinanXie/EGDeblurring) |
|  |
|  |


### Joint Interpolation & Deblurring

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `E-CIR` | [![arXiv](https://img.shields.io/badge/arXiv-2203.01935-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2203.01935)<br>E-CIR: Event-Enhanced Continuous Intensity Recovery | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/chensong1995/E-CIR)](https://github.com/chensong1995/E-CIR) |
| `EVDI` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Unifying_Motion_Deblurring_and_Frame_Interpolation_With_Events_CVPR_2022_paper.pdf)<br>Unifying Motion Deblurring and Frame Interpolation with Events | CVPR 2022 | - | [![GitHub](https://img.shields.io/github/stars/XiangZ-0/EVDI)](https://github.com/XiangZ-0/EVDI) |
| `UniINR` | [![arXiv](https://img.shields.io/badge/arXiv-2305.15078-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2305.15078)<br>UniINR: Event-Guided Unified Rolling Shutter Correction, Deblurring, and Interpolation | ECCV 2024 | - | [![GitHub](https://img.shields.io/github/stars/yunfanLu/UniINR)](https://github.com/yunfanLu/UniINR) |
|  |
|  |


## 3.3 Image Enhancement

### Super-Resolution

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `eSL-Net` | [ECCV Paper](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_10)<br>Event Enhanced High-Quality Image Recovery | ECCV 2020 | - | [![GitHub](https://img.shields.io/github/stars/ShinyWang33/eSL-Net)](https://github.com/ShinyWang33/eSL-Net) |
| `STIR` | [![arXiv](https://img.shields.io/badge/arXiv-2303.13767-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2303.13767)<br>Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution | CVPR 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlis2022.github.io/cvpr23/egvsr) | [![GitHub](https://img.shields.io/github/stars/yunfanLu/INR-Event-VSR)](https://github.com/yunfanLu/INR-Event-VSR) |
| `eSL-Net++` | [![arXiv](https://img.shields.io/badge/arXiv-2302.13766-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2302.13766)<br>Learning to Super-Resolve Blurry Images with Events | TPAMI 2023 | - | [![GitHub](https://img.shields.io/github/stars/gistvision/e2sri)](https://github.com/gistvision/e2sri) |
| `EvTexture` | [![arXiv](https://img.shields.io/badge/arXiv-2406.13457-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.13457)<br>EvTexture: Event-driven Texture Enhancement for Video Super-Resolution | ICML 2024 | - | [![GitHub](https://img.shields.io/github/stars/DachunKai/EvTexture)](https://github.com/DachunKai/EvTexture) |
| `Noise2Image` | [![arXiv](https://img.shields.io/badge/arXiv-2404.01298-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2404.01298)<br>Noise2Image: Noise-Enabled Static Scene Recovery for Event Cameras | Optica 2025 | - | [![GitHub](https://img.shields.io/github/stars/rmcao/noise2image)](https://github.com/rmcao/noise2image) |
|  |
|  |


### HDR Reconstruction

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `NeurImg-HDR` | [Google Drive](https://drive.google.com/file/d/1gfdc9axSIHO3OOIXL6yTCGljj1l54gqO/view)<br>Hybrid High Dynamic Range Imaging Fusing Neuromorphic and Conventional Images | TPAMI 2023 | - | [![GitHub](https://img.shields.io/github/stars/hjynwa/NeurImg-HDR)](https://github.com/hjynwa/NeurImg-HDR) |
| `HDRev-Net` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Learning_Event_Guided_High_Dynamic_Range_Video_Reconstruction_CVPR_2023_paper.html)<br>Learning Event Guided High Dynamic Range Video Reconstruction | CVPR 2023 | - | [![GitHub](https://img.shields.io/github/stars/YixinYang-00/HDRev)](https://github.com/YixinYang-00/HDRev) |
| `EventHDR` | [![arXiv](https://img.shields.io/badge/arXiv-2409.17029-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.17029)<br>EventHDR: From Event to High-Speed HDR Videos and Beyond | TPAMI 2024 | - | - |
|  |
|  |


### Low-Light Enhancement

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EvLowLight` | [ICCV Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liang_Coherent_Event_Guided_Low-Light_Video_Enhancement_ICCV_2023_paper.html)<br>Coherent Event Guided Low-Light Video Enhancement | ICCV 2023 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://sherrycattt.github.io/EvLowLight/) | [![GitHub](https://img.shields.io/github/stars/sherrycattt/EvLowLight)](https://github.com/sherrycattt/EvLowLight) |
| `NER-Net` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Seeing_Motion_at_Nighttime_with_an_Event_Camera_CVPR_2024_paper.html)<br>Seeing Motion at Nighttime with an Event Camera | CVPR 2024 | - | [![GitHub](https://img.shields.io/github/stars/Liu-haoyue/NER-Net)](https://github.com/Liu-haoyue/NER-Net) |
| `EvLight` | [![arXiv](https://img.shields.io/badge/arXiv-2404.00834-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2404.00834)<br>Towards Robust Event-Guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach | CVPR 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlislab22.github.io/eg-lowlight/) | [![GitHub](https://img.shields.io/github/stars/EthanLiang99/EvLight)](https://github.com/EthanLiang99/EvLight) |
| `Sim2Real-EVFI` | [![arXiv](https://img.shields.io/badge/arXiv-2406.08090-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.08090)<br>From Sim-to-Real: Toward General Event-Based Low-Light Frame Interpolation with Per-Scene Optimization | SIGGRAPH Asia 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://openimaginglab.github.io/Sim2Real/) | [![GitHub](https://img.shields.io/github/stars/OpenImagingLab/sim2real)](https://github.com/OpenImagingLab/sim2real) |
| `EvLight++` | [IEEE Paper](https://ieeexplore.ieee.org/document/11192751)<br>EvLight++: Low-Light Video Enhancement With an Event Camera: A Large-Scale Real-World Dataset, Novel Method, and More | TPAMI 2026 | - | [![GitHub](https://img.shields.io/github/stars/EthanLiang99/EvLight)](https://github.com/EthanLiang99/EvLight) |
|  |
|  |


## 3.4 3D Reconstruction

### Geometric & Semi-Dense

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EMVS` | [IJCV Paper](https://rpg.ifi.uzh.ch/docs/IJCV17_Rebecq.pdf)<br>EMVS: Event-based Multi-View Stereo—3D Reconstruction with an Event Camera in Real-Time | IJCV 2018 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_emvs)](https://github.com/uzh-rpg/rpg_emvs) |
| - | [![arXiv](https://img.shields.io/badge/arXiv-1807.07429-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1807.07429)<br>Semi-Dense 3D Reconstruction with a Stereo Event Camera | ECCV 2018 | - | - |
| `EventPS` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_EventPS_Real-Time_Photometric_Stereo_Using_an_Event_Camera_CVPR_2024_paper.pdf)<br>EventPS: Real-Time Photometric Stereo Using an Event Camera | CVPR 2024 | - | [Codeberg](https://codeberg.org/ybh1998/EventPS) |
|  |
|  |


### NeRF-Based

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
| `AE-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2501.02807-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2501.02807)<br>AE-NeRF: Augmenting Event-Based Neural Radiance Fields for Non-ideal Conditions and Larger Scene | AAAI 2025 | - | - |
| `EvHDR-NeRF` | EvHDR-NeRF: Building High Dynamic Range Radiance Fields with Single Exposure Images and Events | AAAI 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://zehaoc.github.io/EvHDR-NeRF/) | [![GitHub](https://img.shields.io/github/stars/Zehaoc/EvHDR-NeRF)](https://github.com/Zehaoc/EvHDR-NeRF) |
| `LSE-NeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2409.06104-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.06104)<br>LSE-NeRF: Learning Sensor Modeling Errors for Deblurred Neural Radiance Fields with RGB-Event Stereo | 3DV 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://ubc-vision.github.io/LSENeRF/) | [![GitHub](https://img.shields.io/github/stars/ubc-vision/LSENeRF)](https://github.com/ubc-vision/LSENeRF) |
| `SaENeRF` | [![arXiv](https://img.shields.io/badge/arXiv-2504.16389-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2504.16389)<br>SaENeRF: Suppressing Artifacts in Event-based Neural Radiance Fields | IJCNN 2025 | - | [![GitHub](https://img.shields.io/github/stars/Mr-firework/SaENeRF)](https://github.com/Mr-firework/SaENeRF) |
| `DynEventNeRF` | Dynamic EventNeRF: Reconstructing General Dynamic Scenes from Multi-view RGB and Event Streams | CVPR Workshop 2025 | - | [![GitHub](https://img.shields.io/github/stars/r00tman/DynEventNeRF)](https://github.com/r00tman/DynEventNeRF) |
| `EventNeuS` | [![arXiv](https://img.shields.io/badge/arXiv-2602.03847-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2602.03847)<br>EventNeuS: 3D Mesh Reconstruction from a Single Event Camera | 3DV 2026 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://4dqv.mpi-inf.mpg.de/EventNeuS/) | - |
|  |
|  |


### 3DGS-Based

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
| `EF-3DGS` | [![arXiv](https://img.shields.io/badge/arXiv-2410.15392-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2410.15392)<br>EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting | NeurIPS 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://lbh666.github.io/ef-3dgs/) | - |
| `E-3DGS` | [![arXiv](https://img.shields.io/badge/arXiv-2410.16995-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2410.16995)<br>E-3DGS: Gaussian Splatting with Exposure and Motion Events | Applied Optics 2025 | - | [![GitHub](https://img.shields.io/github/stars/MasterHow/E-3DGS)](https://github.com/MasterHow/E-3DGS) |
| `EventBoosted-3DGS` | [![arXiv](https://img.shields.io/badge/arXiv-2411.16180-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2411.16180)<br>Event-Boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction | ICCV 2025 | - | - |
| `Event-3DGS` | [NeurIPS Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e73ad1f690542144ce354637bb913c35-Abstract-Conference.html)<br>Event-3DGS: Event-Based 3D Reconstruction Using 3D Gaussian Splatting | NeurIPS 2024 | - | [![GitHub](https://img.shields.io/github/stars/lanpokn/Event-3DGS)](https://github.com/lanpokn/Event-3DGS) |
| `SweepEvGS` | [![arXiv](https://img.shields.io/badge/arXiv-2412.11579-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2412.11579)<br>SweepEvGS: Event-Based 3D Gaussian Splatting for Macro and Micro Radiance Field Rendering from a Single Sweep | TCSVT 2025 | - | - |
| `EventSplat` | [![arXiv](https://img.shields.io/badge/arXiv-2412.07293-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2412.07293)<br>EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-Time Rendering | CVPR 2025 | - | - |
| `BeSplat` | [WACV Paper](https://openaccess.thecvf.com/content/WACV2025W/EVGEN/papers/Matta_BeSplat_Gaussian_Splatting_from_a_Single_Blurry_Image_and_Event_WACVW_2025_paper.pdf)<br>BeSplat: Gaussian Splatting from a Single Blurry Image and Event Stream | WACVW 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://gopirajumatta.github.io/BeSplat/) | [![GitHub](https://img.shields.io/github/stars/GopiRajuMatta/BeSplat)](https://github.com/GopiRajuMatta/BeSplat) |
| `DiET-GS` | [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_DiET-GS_Diffusion_Prior_and_Event_Stream-Assisted_Motion_Deblurring_3D_Gaussian_CVPR_2025_paper.html)<br>DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting | CVPR 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://diet-gs.github.io/) | [![GitHub](https://img.shields.io/github/stars/DiET-GS/DiET-GS)](https://github.com/DiET-GS/DiET-GS) |
| `EBAD-Gaussian` | [![arXiv](https://img.shields.io/badge/arXiv-2504.10012-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2504.10012)<br>EBAD-Gaussian: Event-Driven Bundle Adjusted Deblur Gaussian Splatting | arXiv 2025 | - | - |
| `Elite-EvGS` | [![arXiv](https://img.shields.io/badge/arXiv-2409.13392-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.13392)<br>Elite-EvGS: Learning Event-based 3D Gaussian Splatting by Distilling Event-to-Video Priors | arXiv 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://vlislab22.github.io/elite-evgs/) | - |
|  |
|  |



# 4. Event Camera Understanding

> This section covers methods at the intersection of event cameras and large-scale pre-trained models (LLMs, VLMs, MLLMs, foundation models).


## 4.1 Open-Vocabulary Perception

### Image-Level Recognition

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EventCLIP` | EventCLIP: Adapting CLIP for Event-based Object Recognition | arXiv 2023 | - | - |
| `ExACT` | [![arXiv](https://img.shields.io/badge/arXiv-2403.12534-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.12534)<br>ExACT: Language-guided Conceptual Reasoning and Uncertainty Estimation for Event-based Action Recognition | CVPR 2024 | - | - |
| `EventBind` | EventBind: Learning a Unified Representation to Bind Them All for Event-based Open-world Understanding | ECCV 2024 | - | - |
| `CEIA` | CEIA: CLIP-Based Event-Image Alignment for Open-World Event-Based Understanding | arXiv 2024 | - | - |
| `Expanding Event` | Expanding Event Modality Applications through a Robust CLIP-Based Encoder | arXiv 2024 | - | - |
| `EZSR` | EZSR: Event-based Zero-Shot Recognition | CVPR 2025 | - | - |
|  |
|  |


### Dense Segmentation & Detection

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `OpenESS` | [![arXiv](https://img.shields.io/badge/arXiv-2405.05259-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2405.05259)<br>OpenESS: Event-Based Semantic Scene Understanding with Open Vocabularies | CVPR 2024 (Highlight) | - | [![GitHub](https://img.shields.io/github/stars/ldkong1205/OpenESS)](https://github.com/ldkong1205/OpenESS) |
| `EventSAM` | [![arXiv](https://img.shields.io/badge/arXiv-2312.16222-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2312.16222)<br>Segment Any Event Streams via Weighted Adaptation of Pivotal Tokens | CVPR 2024 | - | [![GitHub](https://img.shields.io/github/stars/zhiwen-xdu/EventSAM)](https://github.com/zhiwen-xdu/EventSAM) |
| `SAM-Event-Adapter` | SAM-Event-Adapter: Adapting Segment Anything Model for Event-RGB Semantic Segmentation | ICRA 2024 | - | - |
| `OVOSE` | [![arXiv](https://img.shields.io/badge/arXiv-2408.09424-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2408.09424)<br>OVOSE: Open-Vocabulary Semantic Segmentation in Event-Based Cameras | arXiv 2024 | - | - |
| `OV-Detection` | Adaptive Event Stream Slicing for Open-Vocabulary Event-Based Object Detection | arXiv 2025 | - | - |
| `EvSAM` | EvSAM: Segment Anything Model with Event-based Assistance | ACM TOMM 2025 | - | - |
| `SEAL` | [![arXiv](https://img.shields.io/badge/arXiv-2601.23159-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2601.23159)<br>Segment Any Events with Language | ICLR 2026 | - | - |
|  |
|  |


### Self-Supervised Pre-training

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `ECDP` | [![arXiv](https://img.shields.io/badge/arXiv-2301.01928-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2301.01928)<br>Event Camera Data Pre-training | ICCV 2023 | - | [![GitHub](https://img.shields.io/github/stars/Yan98/Event-Camera-Data-Pre-training)](https://github.com/Yan98/Event-Camera-Data-Pre-training) |
| `MEM` | [![arXiv](https://img.shields.io/badge/arXiv-2212.10368-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2212.10368)<br>Masked Event Modeling: Self-Supervised Pretraining for Event Cameras | WACV 2024 | - | [![GitHub](https://img.shields.io/github/stars/tum-vision/mem)](https://github.com/tum-vision/mem) |
| `DECP` | Data-efficient Event Camera Pre-training via Disentangled Masked Modeling | arXiv 2024 | - | - |
| `ECDP-Dense` | Event Camera Data Dense Pre-training | ECCV 2024 | - | - |
| `SD2Event` | SD2Event: Self-supervised Learning of Dynamic Detectors and Contextual Descriptors for Event Cameras | CVPR 2024 | - | - |
| `EvRepSL` | Event-Stream Representation via Self-Supervised Learning for Event-Based Vision | TIP 2024 | - | - |
| `TESPEC` | Temporally-Enhanced Self-Supervised Pretraining for Event Cameras | ICCV 2025 | - | - |
| `STP` | Efficient Event Camera Data Pretraining with Adaptive Prompt Fusion | ICCV 2025 | - | - |
| `EventPretrain` | Revealing Latent Information: A Physics-inspired Self-supervised Pre-training Framework | ACM MM 2025 | - | - |
| `CM3AE` | A Unified RGB Frame and Event-Voxel/-Frame Pre-training Framework | ACM MM 2025 | - | - |
|  |
|  |


### Cross-Modal Transfer

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `EventDance` | EventDance: Unsupervised Source-free Cross-modal Adaptation for Event-based Object Recognition | CVPR 2024 | - | - |
| `S5-ViT` | State Space Models for Event Cameras | CVPR 2024 (Spotlight) | - | - |
| `Spike-DINOv2` | A Novel Energy-Efficient Spike Transformer Network for Depth Estimation via Cross-modality Knowledge Distillation | arXiv 2024 | - | - |
| `EventFly` | EventFly: Event Camera Perception from Ground to the Sky | CVPR 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://event-fly.github.io/) | - |
| `Depth AnyEvent` | Depth Any Event Stream: Enhancing Event-based Monocular Depth Estimation via Dense-to-Sparse Distillation | ICCV 2025 | - | - |
| `FFEvent` | FFEvent: Fast Fourier-based Knowledge Transfer for Event Cameras | Expert Systems with Applications 2026 | - | - |
| `Semantic-E2VID` | Exploring The Missing Semantics In Event Modality | arXiv 2025 | - | - |
| `TGVFM` | Temporal-Guided Visual Foundation Models for Event-Based Vision | arXiv 2025 | - | - |
|  |
|  |


## 4.2 Scene Understanding & Reasoning

### Event-Based MLLMs

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `LLM Zero-Shot` | Can Large Language Models Grasp Event Signals? Exploring Pure Zero-Shot Event-based Recognition | ICASSP 2025 | - | - |
| `EventGPT` | EventGPT: Event Stream Understanding with Multimodal Large Language Models | CVPR 2025 | - | - |
| `LLM-EvRep` | Learning an LLM-Compatible Event Representation Using a Self-Supervised Framework | WWW Companion 2025 | - | - |
| `EventVL` | EventVL: Understand Event Streams via Multimodal Large Language Model | arXiv 2025 | - | - |
| `EventFlash` | EventFlash: Towards Efficient MLLMs for Event-Based Vision | arXiv 2026 | - | - |
|  |
|  |


### Visual Grounding & Embodied Intelligence

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `NeuroViG` | NeuroViG: Integrating Event Cameras for Resource-Efficient Video Grounding | WACV 2025 | - | - |
| `EP-VLM` | Event-Priori-Based Vision-Language Model for Efficient Visual Understanding | arXiv 2025 | - | - |
| `Talk2Event` | Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras | NeurIPS 2025 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://talk2event.github.io/) | - |
| `E-VLA` | E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes | arXiv 2026 | - | - |
|  |
|  |


### Language-Guided Recognition

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `Ev-LaFOR` | Label-Free Event-based Object Recognition via Joint Learning with Image Reconstruction from Events | ICCV 2023 (Oral) | - | - |
| `EventDance++` | EventDance++: Language-guided Unsupervised Source-free Cross-modal Adaptation for Event-based Object Recognition | arXiv 2024 | - | - |
| `ExACT` | [![arXiv](https://img.shields.io/badge/arXiv-2403.12534-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.12534)<br>ExACT: Language-guided Conceptual Reasoning and Uncertainty Estimation for Event-based Action Recognition | CVPR 2024 | - | - |
|  |
|  |


## 4.3 Event Data Simulation & Generation

### Physics-Based Simulators

> :timer_clock: In chronological order, from the earliest to the latest.

| Simulator | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `ESIM` | [ESIM: an Open Event Camera Simulator](https://rpg.ifi.uzh.ch/esim.html) | CoRL 2018 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_esim)](https://github.com/uzh-rpg/rpg_esim) |
| `rpg_vid2e` | [![arXiv](https://img.shields.io/badge/arXiv-1912.03095-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1912.03095)<br>Video to Events: Recycling Video Datasets for Event Cameras | CVPR 2020 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_vid2e)](https://github.com/uzh-rpg/rpg_vid2e) |
| `v2e` | [![arXiv](https://img.shields.io/badge/arXiv-2006.07722-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2006.07722)<br>v2e: From Video Frames to Realistic DVS Events | CVPRW 2021 | - | [![GitHub](https://img.shields.io/github/stars/SensorsINI/v2e)](https://github.com/SensorsINI/v2e) |
| `DVS-Voltmeter` | [DVS-Voltmeter: Stochastic Process-Based Event Simulator for Dynamic Vision Sensors](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_34) | ECCV 2022 | - | [![GitHub](https://img.shields.io/github/stars/Lynn0306/DVS-Voltmeter)](https://github.com/Lynn0306/DVS-Voltmeter) |
|  |
|  |


### Learned & Neural Generation

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `Text-to-Events` | [![arXiv](https://img.shields.io/badge/arXiv-2406.03439-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.03439)<br>Text-to-Events: Synthetic Event Camera Streams from Conditional Text Input | NICE 2024 | - | - |
| `ControlEvents` | [![arXiv](https://img.shields.io/badge/arXiv-2509.22864-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2509.22864)<br>ControlEvents: Controllable Synthesis of Event Camera Data with Foundational Prior from Image Diffusion Models | WACV 2026 | - | - |
| `ShapeAug` | [![arXiv](https://img.shields.io/badge/arXiv-2409.11075-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.11075)<br>ShapeAug++: More Realistic Shape Augmentation for Event Data | arXiv 2024 | - | - |
|  |
|  |


### Sim-to-Real Transfer

> :timer_clock: In chronological order, from the earliest to the latest.

| Model | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `E2VID+` | [![arXiv](https://img.shields.io/badge/arXiv-2003.09078-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2003.09078)<br>Reducing the Sim-to-Real Gap for Event Cameras | ECCV 2020 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://timostoff.github.io/20ecnn) | [![GitHub](https://img.shields.io/github/stars/TimoStoff/event_cnn_minimal)](https://github.com/TimoStoff/event_cnn_minimal) |
| `DA4Event` | DA4Event: Towards Bridging the Sim-to-Real Gap for Event Cameras Using Domain Adaptation | arXiv 2021 | - | - |
| `N-ROD` | N-ROD: A Neuromorphic Dataset for Synthetic-to-Real Domain Adaptation | arXiv 2021 | - | - |
| `Sim2Real-EVFI` | [![arXiv](https://img.shields.io/badge/arXiv-2406.08090-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2406.08090)<br>From Sim-to-Real: Toward General Event-Based Low-Light Frame Interpolation with Per-Scene Optimization | SIGGRAPH Asia 2024 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://openimaginglab.github.io/Sim2Real/) | [![GitHub](https://img.shields.io/github/stars/OpenImagingLab/sim2real)](https://github.com/OpenImagingLab/sim2real) |
|  |
|  |



# 5. Datasets & Benchmarks

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

#### Classification Datasets

| Dataset | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `N-MNIST` | [![arXiv](https://img.shields.io/badge/arXiv-1507.07629-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1507.07629)<br>Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades | Front. Neurosci. 2015 | - | - |
| `N-Caltech101` | [![arXiv](https://img.shields.io/badge/arXiv-1507.07629-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1507.07629)<br>Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades | Front. Neurosci. 2015 | - | - |
| `CIFAR10-DVS` | CIFAR10-DVS: An Event-Stream Dataset for Object Classification | Front. Neurosci. 2017 | - | - |
| `N-Cars` | [![arXiv](https://img.shields.io/badge/arXiv-1803.07913-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1803.07913)<br>HATS: Histograms of Averaged Time Surfaces for Robust Event-based Object Classification | CVPR 2018 | - | - |
| `N-ImageNet` | [![arXiv](https://img.shields.io/badge/arXiv-2112.01041-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2112.01041)<br>N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras | ICCV 2021 | - | [![GitHub](https://img.shields.io/github/stars/82magnolia/n_imagenet)](https://github.com/82magnolia/n_imagenet) |
| `ES-ImageNet` | [![arXiv](https://img.shields.io/badge/arXiv-2110.12211-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2110.12211)<br>ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks | Front. Neurosci. 2021 | - | - |
|  |
|  |

#### Object Detection Datasets

| Dataset | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `Gen1` | [![arXiv](https://img.shields.io/badge/arXiv-2001.08499-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2001.08499)<br>A Large Scale Event-based Detection Dataset for Automotive | arXiv 2020 | - | - |
| `1Mpx` | [![arXiv](https://img.shields.io/badge/arXiv-2009.13436-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2009.13436)<br>Learning to Detect Objects with a 1 Megapixel Event Camera | NeurIPS 2020 | - | - |
| `PEDRo` | [PEDRo: an Event-based Dataset for Person Detection in Robotics](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/html/Boretti_PEDRo_An_Event-Based_Dataset_for_Person_Detection_in_Robotics_CVPRW_2023_paper.html) | CVPRW 2023 | - | - |
| `PKU-DAVIS-SOD` | [![arXiv](https://img.shields.io/badge/arXiv-2308.04047-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2308.04047)<br>SODFormer: Streaming Object Detection with Transformer Using Events and Frames | TPAMI 2023 | - | - |
| `EvDET200K` | Object Detection using Event Camera: A MoE Heat Conduction based Detector | CVPR 2025 | - | [![GitHub stars](https://img.shields.io/github/stars/Event-AHU/OpenEvDET?style=flat-square&logo=github)](https://github.com/Event-AHU/OpenEvDET) |

#### Object Tracking Datasets

| Dataset | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `FE108` | [![arXiv](https://img.shields.io/badge/arXiv-2109.09052-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2109.09052)<br>Object Tracking by Jointly Exploiting Frame and Event Domain | ICCV 2021 | - | [![GitHub stars](https://img.shields.io/github/stars/Jee-King/ICCV2021_Event_Frame_Tracking?style=flat-square&logo=github)](https://github.com/Jee-King/ICCV2021_Event_Frame_Tracking) |
| `VisEvent` | [![arXiv](https://img.shields.io/badge/arXiv-2108.05015-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2108.05015)<br>VisEvent: Reliable Object Tracking via Collaboration of Frame and Event Flows | TCYB 2024 | - | [![GitHub stars](https://img.shields.io/github/stars/wangxiao5791509/VisEvent_SOT_Benchmark?style=flat-square&logo=github)](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark) |
| `COESOT` | [![arXiv](https://img.shields.io/badge/arXiv-2211.11010-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2211.11010)<br>Revisiting Color-Event based Tracking: A Unified Network, Dataset, and Metric | PR 2025 | - | [![GitHub stars](https://img.shields.io/github/stars/Event-AHU/COESOT?style=flat-square&logo=github)](https://github.com/Event-AHU/COESOT) |
| `EventVOT` | [![arXiv](https://img.shields.io/badge/arXiv-2309.14611-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2309.14611)<br>Event Stream-based Visual Object Tracking: A High-Resolution Benchmark Dataset and A Novel Baseline | CVPR 2024 | - | [![GitHub stars](https://img.shields.io/github/stars/Event-AHU/EventVOT_Benchmark?style=flat-square&logo=github)](https://github.com/Event-AHU/EventVOT_Benchmark) |
| `FELT` | [![arXiv](https://img.shields.io/badge/arXiv-2409.05765-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.05765)<br>Long-Term Visual Object Tracking with Event Cameras: An Associative Memory Augmented Tracker and A Benchmark Dataset | arXiv 2025 | - | [![GitHub stars](https://img.shields.io/github/stars/Event-AHU/FELT_SOT_Benchmark?style=flat-square&logo=github)](https://github.com/Event-AHU/FELT_SOT_Benchmark/tree/main) |
| `CRSOT` | [![arXiv](https://img.shields.io/badge/arXiv-2401.02826-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2401.02826)<br>Cross-Resolution Object Tracking Using Unaligned Frame and Event Cameras | TMM 2025 | - | [![GitHub stars](https://img.shields.io/github/stars/Event-AHU/Cross_Resolution_SOT?style=flat-square&logo=github)](https://github.com/Event-AHU/Cross_Resolution_SOT) |

#### General / Multi-Task Datasets

| Dataset | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `DSEC` | [![arXiv](https://img.shields.io/badge/arXiv-2103.06011-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2103.06011)<br>DSEC: A Stereo Event Camera Dataset for Driving Scenarios | RA-L 2021 | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://dsec.ifi.uzh.ch/) | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/DSEC)](https://github.com/uzh-rpg/DSEC) |
| `DSEC-Semantic` | ESS: Learning Event-based Semantic Segmentation from Still Images | ECCV 2022 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/ess)](https://github.com/uzh-rpg/ess) |
|  |
|  |


### Simulators

| Simulator | Paper | Venue | Website | GitHub |
|:-:|:-|:-:|:-:|:-:|
| `ESIM` | [ESIM: an Open Event Camera Simulator](https://rpg.ifi.uzh.ch/esim.html) | CoRL 2018 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_esim)](https://github.com/uzh-rpg/rpg_esim) |
| `rpg_vid2e` | [![arXiv](https://img.shields.io/badge/arXiv-1912.03095-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1912.03095)<br>Video to Events: Recycling Video Datasets for Event Cameras | CVPR 2020 | - | [![GitHub](https://img.shields.io/github/stars/uzh-rpg/rpg_vid2e)](https://github.com/uzh-rpg/rpg_vid2e) |
| `v2e` | [![arXiv](https://img.shields.io/badge/arXiv-2006.07722-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2006.07722)<br>v2e: From Video Frames to Realistic DVS Events | CVPRW 2021 | - | [![GitHub](https://img.shields.io/github/stars/SensorsINI/v2e)](https://github.com/SensorsINI/v2e) |
| `DVS-Voltmeter` | [DVS-Voltmeter: Stochastic Process-Based Event Simulator for Dynamic Vision Sensors](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_34) | ECCV 2022 | - | [![GitHub](https://img.shields.io/github/stars/Lynn0306/DVS-Voltmeter)](https://github.com/Lynn0306/DVS-Voltmeter) |
|  |
|  |



# 6. Challenges & Future Directions

...



# 7. Other Resources

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



# 8. Acknowledgements

To be added.
