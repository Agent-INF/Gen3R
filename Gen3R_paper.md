# <span id="page-0-0"></span>Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction

Jiaxin Huang<sup>1</sup> , Yuanbo Yang<sup>1</sup> , Bangbang Yang<sup>2</sup> , Lin Ma<sup>2</sup> , Yuewen Ma<sup>2</sup> , Yiyi Liao<sup>1</sup><sup>B</sup> <sup>1</sup> Zhejiang University <sup>2</sup> ByteDance

Project Page: <https://xdimlab.github.io/Gen3R/>

![](_page_0_Picture_4.jpeg)

Figure 1. Gen3R bridges foundational reconstruction models with 2D video diffusion, enabling the joint generation of 2D videos and their corresponding geometry in various settings.

# Abstract

*We present Gen3R, a method that bridges the strong priors of foundational reconstruction models and video diffusion models for scene-level 3D generation. We repurpose the VGGT reconstruction model to produce geometric latents by training an adapter on its tokens, which are regularized to align with the appearance latents of pre-trained video diffusion models. By jointly generating these disentangled yet aligned latents, Gen3R produces both RGB videos and corresponding 3D geometry, including camera poses, depth maps, and global point clouds. Experiments demonstrate that our approach achieves state-of-the-art results in singleand multi-image conditioned 3D scene generation. Additionally, our method can enhance the robustness of reconstruction by leveraging generative priors, demonstrating the mutual benefit of tightly coupling reconstruction and* *generative models.*

# 1. Introduction

3D scene generation has become a fundamental problem in computer vision and graphics, with wide applications in simulation, gaming, robotics, and virtual reality. A method capable of producing photorealistic and geometrically consistent 3D scenes would enable the creation of immersive environments at scale, serving as essential training data and providing new tools for creative content design.

Prior methods attempt to extend 2D generative models via score distillation [\[29,](#page-9-0) [40,](#page-9-1) [57,](#page-10-0) [71\]](#page-10-1), incremental outpainting [\[6,](#page-8-0) [9,](#page-8-1) [81,](#page-11-0) [82\]](#page-11-1), or multi-view synthesis followed by reconstruction [\[10,](#page-8-2) [12,](#page-8-3) [33,](#page-9-2) [47,](#page-9-3) [53,](#page-10-2) [74\]](#page-10-3). Despite promising results, these methods often suffer from poor geometric structure or high optimization cost. More recently, several works [\[11,](#page-8-4) [24,](#page-9-4) [56,](#page-10-4) [79,](#page-11-2) [87\]](#page-11-3) have extended video diffusion

<sup>B</sup> Corresponding author.

<span id="page-1-0"></span>frameworks to *feed-forward 3D scene generation* for improved efficiency. These approaches typically follow the Latent Diffusion Model paradigm, training a VAE to learn a compact latent space for 3D scenes and applying diffusion within that space. However, the scarcity of large-scale 3D ground truth makes learning geometry-centric VAEs highly challenging. One line of methods trains a VAE to reconstruct geometry from RGB inputs while simultaneously learning a compressed latent representation [\[24,](#page-9-4) [49,](#page-10-5) [79\]](#page-11-2). Yet this is inherently difficult, especially when supervision is limited to 2D signals, which often results in suboptimal geometry and constrained generation quality.

In parallel, transformer-based feed-forward reconstruction models, such as Dust3R [\[67\]](#page-10-6) and VGGT [\[65\]](#page-10-7), have shown strong reconstruction ability from 2D images. Recent works attempt to build better VAEs by compressing their 3D output [\[56,](#page-10-4) [87\]](#page-11-3), but overlook a key fact: these reconstruction models already operate in a spatially compact token space that encodes rich multi-view geometric information, including depth, camera pose, and global structure. This observation raises a central question: Can the intrinsic latent manifold learned by reconstruction models be used to fully exploit reconstruction priors for 3D scene generation?

Building on this insight, we introduce Gen3R, a 3Daware scene generation method that unifies advanced reconstruction and generation models for jointly generating controllable video and globally consistent 3D point clouds. Our key idea is to recast a feed-forward reconstruction model, VGGT [\[65\]](#page-10-7), as a VAE-like provider of geometric latents and combine these with appearance latents from a pre-trained video diffusion model for joint generation. This allows us to marry the rich geometric priors learned by reconstruction models over multiple 3D quantities with the strong RGB priors of video diffusion models, effectively combining the strengths of both. To achieve so, we first project the reconstruction model's intermediate tokens to match the spatial-temporal resolution of the appearance latents using a learned adapter. Notably, simply compressing the tokens is insufficient as their distribution significantly differs from the corresponding appearance latents. We therefore propose to align the two latent spaces, followed by fine-tuning a video diffusion model for joint generation. By keeping geometric and appearance latents disentangled while aligning their distributions, Gen3R demonstrates that the latent manifold learned by reconstruction models can indeed serve as a strong foundation for high-fidelity 3D scene generation.

Our framework supports flexible conditioning, enabling generation from single or multiple input views, with or without camera cues, as well as feed-forward scene reconstruction within one unified model. It produces temporally coherent RGB videos and globally aligned point clouds across diverse configurations.

Our contributions are threefold: 1) A novel framework

integrating video diffusion models with geometric foundation models, combining strong RGB priors with rich geometric priors for 3D scene generation. 2) A disentangled yet aligned appearance and geometry latent space, enabling controllable and multi-view consistent scene synthesis. 3) A flexible pipeline capable of handling various input settings, producing high-fidelity videos and globally consistent 3D point clouds.

# 2. Related Work

3D Scene Generation fom 2D Priors. A common strategy for 3D scene generation is to leverage pretrained 2D generative models [\[46\]](#page-9-5) to provide RGB priors. One line of work employs score distillation sampling (SDS) [\[29,](#page-9-0) [40,](#page-9-1) [57,](#page-10-0) [71\]](#page-10-1), directly optimizing a 3D representation such as NeRF [\[36\]](#page-9-6) and 3DGS [\[21\]](#page-9-7) to align with the distribution of a 2D diffusion model. Another line of methods first synthesizes multi-view images using pretrained 2D diffusion models, followed by 3D reconstruction through multi-view synthesis [\[4,](#page-8-5) [10,](#page-8-2) [12,](#page-8-3) [32,](#page-9-8) [33,](#page-9-2) [47,](#page-9-3) [53,](#page-10-2) [55,](#page-10-8) [74,](#page-10-3) [89\]](#page-11-4) or incremental outpainting [\[6,](#page-8-0) [9,](#page-8-1) [51,](#page-10-9) [81,](#page-11-0) [82\]](#page-11-1). Both paradigms leverage the strong RGB priors of 2D models but are limited by the lack of explicit 3D reasoning, often resulting in inconsistent geometry, weak multi-view fidelity, and high computational cost. Our method tackles this challenge by bridging rich geometric priors of a reconstruction foundation model with a 2D generative model.

Feed-Forward 3D Scene Generation. Object-level feedforward 3D generation methods [\[28,](#page-9-9) [35,](#page-9-10) [45,](#page-9-11) [76,](#page-11-5) [86\]](#page-11-6) have gained great success thanks to the large-scale 3D ground truth datasets [\[7,](#page-8-6) [27\]](#page-9-12). However, extending this success to full-scene 3D generation is challenging because highquality scene-level data is difficult to obtain. A practical alternative is to synthesize a 3D representation in a feedforward manner and train it using only 2D supervision. Recent works [\[11,](#page-8-4) [24,](#page-9-4) [25,](#page-9-13) [38,](#page-9-14) [50,](#page-10-10) [72,](#page-10-11) [79\]](#page-11-2) follow this strategy by generating Gaussians and using differentiable rendering to train directly on 2D images, thereby avoiding costly 3D data collection. However, these methods often struggle with intricate geometric details and multi-view consistency due to the lack of explicit 3D supervision. Other approaches [\[39,](#page-9-15) [56,](#page-10-4) [59,](#page-10-12) [87\]](#page-11-3) address this limitation by leveraging off-the-shelf dense reconstruction models [\[22,](#page-9-16) [67\]](#page-10-6) or Unreal Engine to obtain 3D data for training. In contrast to methods directly compressing the 3D output of reconstruction models [\[59,](#page-10-12) [87\]](#page-11-3), our approach treats the reconstruction model as an asymmetric VAE that encodes images into geometry latents, allowing us to inherit the strong geometric priors across multiple 3D quantities and the high-level scene understanding from the foundation geometry model.

Feed-forward 3D Scene Reconstruction. Traditional 3D

<span id="page-2-2"></span><span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 2. **Method.** Left: We recast an advanced transformer-based feed-forward reconstruction model, VGGT, as a VAE to produce geometry latents  $\mathcal{G}$  by training an adapter on its latent tokens. The training is supervised with a reconstruction loss  $\mathcal{L}_{rec}$ , along with a regularization term  $\mathcal{L}_{KL}$  that aligns  $\mathcal{G}$  with the appearance latent  $\mathcal{A}$ , which is obtained from the VAE of a pre-trained video diffusion model, WAN. Right: We fine-tune the video diffusion model to jointly generate geometry and appearance latents,  $\mathcal{Z} = [\mathcal{A}; \mathcal{G}]$ , under various conditioning signals. At inference, varying the conditioning enables the generation of RGB videos and multiple 3D quantities, including global point clouds, depth maps, and camera parameters, from a single or multiple frames, as well as performing reconstruction.

scene reconstruction pipelines [3, 37, 48, 60] have recently been complemented by learning-based methods [15, 22, 58, 64, 66, 67, 84, 85], which leverage neural architectures to capture structural regularities of the world. Among these, seminal works such as [22, 67] demonstrated the ability to infer geometrically consistent point clouds from uncalibrated images. Recent advances, exemplified by [20, 65], provide unified frameworks that jointly estimate camera parameters, dense geometry and point tracks. Subsequent studies have further extended VGGT to new scene representations [18, 23, 34, 63] or addressed its inherent limitations [52, 69, 91].

Our method integrates the geometric prior of such feedforward reconstruction models [65] with a generative diffusion model [14, 62, 80]. Different from prior reconstruction approaches [18, 52, 65, 67, 69], our method is inherently generative, capable of synthesizing coherent 3D scenes from 1 or 2 views. Moreover, our method can also be used for performing reconstruction and is able to mitigate errors of the original reconstruction model.

### 3. Method

Our goal is to generate high-fidelity 3D scenes with consistent geometry and controllable cameras given one or more images. To achieve this, we propose Gen3R, a 3D-aware latent diffusion method bridging foundational reconstruction models with pre-trained video diffusion models.

Specifically, we first design a unified latent space for appearance and geometry by recasting the geometry features of the feed-forward reconstruction model, VGGT, into the latent space of a video diffusion model (Sec. 3.1). We then fine-tune the video diffusion model to jointly generate the appearance and geometry latents under various conditions (Sec. 3.2). Finally, these latents are decoded sep-

arately into RGB frames and scene geometry, including global point clouds, depth maps and camera parameters (Sec. 3.3). Fig. 2 illustrates our overall architecture.

### <span id="page-2-0"></span>3.1. Geometry Adapter for Unified Latent Space

**Preliminary.** VGGT [65], denoted as  $\mathcal{F}$ , is a transformer-based feed-forward reconstruction model that directly infers multiple key 3D quantities of a scene from observed views. It takes N input images  $\mathcal{I} \in \mathbb{R}^{N \times H \times W \times 3}$  and encodes them into high-dimensional geometry tokens  $\mathcal{V} \in \mathbb{R}^{N \times h_v \times w_v \times C}$  by its encoder  $\mathcal{E}_{\mathcal{V}}$ , which consists of 24 attention blocks:

$$\mathcal{E}_{\mathcal{V}}: \mathcal{I} \to \mathcal{V} \in \mathbb{R}^{N \times L \times h_v \times w_v \times C}, \tag{1}$$

where  $h_v \times w_v$  is the token's spatial resolution, C=2048, and L=4 is the number of intermediate transformer tokens—specifically those from the 4th, 11th, 17th and 23rd blocks [78]—for subsequent decoding. For simplicity, we omit the camera tokens of VGGT in the description; please refer to Supplementary for details.

The geometry tokens  $\mathcal{V}$  are then decoded by several individual DPT heads [42]  $\mathcal{D}_{\mathcal{V}}$  into multi-modal dense predictions, such as point clouds  $\mathcal{P} \in \mathbb{R}^{N \times H \times W \times 3}$ , depth maps  $\mathcal{D} \in \mathbb{R}^{N \times H \times W \times 1}$  and camera parameters  $\mathcal{T} \in \mathbb{R}^{N \times 9}$ :

$$\mathcal{D}_{\mathcal{V}}: \mathcal{V} \to (\mathcal{P}, \mathcal{D}, \mathcal{T}).$$
 (2)

**Token-to-Latent Adapter.** We aim to recast VGGT [65] as an asymmetric geometry VAE that takes as input N RGB images  $\mathcal{I} \in \mathbb{R}^{N \times H \times W \times 3}$ , produces geometric latents  $\mathcal{G} \in \mathbb{R}^{n \times h \times w \times c}$  for diffusion-based generation, and decodes them into multi-modal geometric outputs, including globally consistent point clouds  $\hat{\mathcal{P}} \in \mathbb{R}^{N \times H \times W \times 3}$ , perview depth maps  $\hat{\mathcal{D}} \in \mathbb{R}^{N \times H \times W \times 1}$  and camera parame-

<span id="page-3-2"></span><span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 3. Qualitative Comparison of Geometry Generation in the 1-view based setting.

ters  $\hat{\mathcal{T}} \in \mathbb{R}^{N \times 9}$ . Since the latent space of a video diffusion model exhibits a different spatial-temporal resolution from VGGT tokens and operates in a substantially lower-dimensional feature space (e.g., c=16 [62]), we train an adapter ( $\mathcal{E}_{\mathrm{adp}}, \mathcal{D}_{\mathrm{adp}}$ ) to bridge the two feature spaces by mapping the geometric tokens  $\mathcal{V}$  into the latent space of the video diffusion model and project them back:

$$\mathcal{E}_{\text{adp}}: \mathcal{V} \to \mathcal{G} \in \mathbb{R}^{n \times h \times w \times c},$$
 (3)

$$\mathcal{D}_{\text{adp}}: \mathcal{G} \to \mathcal{V} \in \mathbb{R}^{N \times L \times h_v \times w_v \times C}, \tag{4}$$

where  $n \times h \times w$  is the downsampled resolution.

The resulting geometric latents  $\mathcal{G}$  share the same spatial-temporal resolution and feature dimension as those of the video diffusion model, enabling joint generation of appearance and geometry within a unified latent space.

**Training of the Adapter.** Our adapter is trained with a reconstruction loss and a distribution alignment loss wrt. the appearance latents:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{rec}} + \lambda_2 \mathcal{L}_{\text{KL}}. \tag{5}$$

Specifically, the reconstruction loss enforces the reconstructed geometry tokens  $\hat{\mathcal{V}} = \mathcal{D}_{\mathrm{adp}}(\mathcal{G})$  to match the original tokens  $\mathcal{V}$ , and further regularize the consistency between the decoded outputs  $(\hat{\mathcal{P}}, \hat{\mathcal{D}}, \hat{\mathcal{T}})$  and those derived from the original tokens  $(\mathcal{P}, \mathcal{D}, \mathcal{T})$  by the pretrained DPT heads:

$$\mathcal{L}_{rec} = \mathbb{E}[\|\hat{\mathcal{V}} - \mathcal{V}\|^2] + \mathbb{E}[\|\hat{\mathcal{T}} - \mathcal{T}\|_1] + \mathbb{E}[\|\hat{\mathcal{D}} - \mathcal{D}\|^2] + \mathbb{E}[\|\hat{\mathcal{P}} - \mathcal{P}\|^2].$$
 (6)

Furthermore, we observed in practice that although this supervision alone effectively compresses the geometry tokens, it does not constrain the mapped latent space, which hinders diffusion training from converging and degrades generation quality. While most LDMs use VAE or VQ-VAE to constrain the latents, we propose to directly regularize our latent space by aligning it with the pretrained appearance latent distribution. Specifically, we impose a KL loss on the geometry adapter, encouraging its latent distribution  $q_{\mathcal{G}}$  to align with the pretrained RGB latent distribution  $q_{\mathcal{A}}$ :

$$\mathcal{L}_{KL} = D_{KL}(q_{\mathcal{G}} \parallel q_{\mathcal{A}}). \tag{7}$$

This constraint ensures compatibility between the two latent spaces and facilitates simultaneous modeling of both appearance and geometry distributions.

### <span id="page-3-0"></span>3.2. Geometry-Aware Joint Latent Generation

**Design of the Joint Latent Space.** After training of the adapter, we establish a compact latent space where geometry and appearance latents can be jointly processed. We then fine-tune a video diffusion model [62], denoted as  $G_{\theta}$ , to generate both modalities of latents within this unified space.

Specifically, we aim to generate latent codes  $\mathcal Z$  consisting of two components: appearance latents  $\mathcal A = \mathcal E_{\mathcal W}(\mathcal I) \in \mathbb R^{n \times h \times w \times c}$  and geometry latents  $\mathcal G \in \mathbb R^{n \times h \times w \times c}$ . To avoid introducing additional trainable parameters and to preserve the pretrained video diffusion model's generative capability, we concatenate the two latents along the width dimension [5] to form a unified latent representation:

$$\mathcal{Z} = [\mathcal{A}; \mathcal{G}] \in \mathbb{R}^{n \times h \times 2w \times c}, \tag{8}$$

where  $[\cdot;\cdot]$  denotes concatenation in the width dimension.

**Training of the Diffusion Model.** To enhance controllability, we incorporate multiple condition signals into the

<span id="page-4-3"></span><span id="page-4-2"></span>

| Cond.  | Method          |        |        |         | RealEstate10K |          |        |        |        |         | DL3DV-10K   |          |        |
|--------|-----------------|--------|--------|---------|---------------|----------|--------|--------|--------|---------|-------------|----------|--------|
|        |                 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | I2V Subj. ↑   | I2V BG ↑ | I.Q. ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | I2V Subj. ↑ | I2V BG ↑ | I.Q. ↑ |
|        | LVSM [19]       | 18.97  | 0.7161 | 0.2992  | 0.9946        | 0.9933   | 0.4923 | 15.61  | 0.5384 | 0.4434  | 0.9635      | 0.9675   | 0.4262 |
|        | Gen3C [44]      | 20.26  | 0.7186 | 0.2302  | 0.9931        | 0.9927   | 0.5200 | 16.21  | 0.5557 | 0.4575  | 0.9427      | 0.9547   | 0.4204 |
| 1-view | GF [73]         | 16.32  | 0.5434 | 0.3803  | 0.9882        | 0.9789   | 0.5614 | 12.05  | 0.3458 | 0.5801  | 0.9335      | 0.9307   | 0.5410 |
|        | Aether [59]     | 16.57  | 0.6374 | 0.3808  | 0.9927        | 0.9910   | 0.5419 | 13.82  | 0.5167 | 0.5272  | 0.9589      | 0.9653   | 0.4571 |
|        | WVD [87]        | 17.62  | 0.6658 | 0.3300  | 0.9935        | 0.9932   | 0.5847 | 14.25  | 0.4848 | 0.5063  | 0.9531      | 0.9613   | 0.5466 |
|        | Ours            | 20.51  | 0.7388 | 0.2281  | 0.9951        | 0.9952   | 0.5993 | 16.38  | 0.5821 | 0.4234  | 0.9657      | 0.9715   | 0.5497 |
|        | DepthSplat [77] | 26.67  | 0.8711 | 0.1742  | 0.9909        | 0.9867   | 0.4379 | 16.83  | 0.6094 | 0.3971  | 0.9505      | 0.9532   | 0.3855 |
|        | LVSM [19]       | 29.58  | 0.9197 | 0.1060  | 0.9954        | 0.9943   | 0.5173 | 18.80  | 0.6404 | 0.3575  | 0.9704      | 0.9736   | 0.4616 |
|        | Gen3C [44]      | 23.83  | 0.8340 | 0.1947  | 0.9936        | 0.9930   | 0.5191 | 17.91  | 0.6120 | 0.4207  | 0.9470      | 0.9566   | 0.4239 |
| 2-view | GF [73]         | 23.28  | 0.7426 | 0.2098  | 0.9893        | 0.9798   | 0.5614 | 14.39  | 0.4152 | 0.5160  | 0.9050      | 0.9110   | 0.4911 |
|        | Aether [59]     | 21.77  | 0.7645 | 0.2241  | 0.9919        | 0.9901   | 0.5258 | 15.68  | 0.5565 | 0.4555  | 0.9619      | 0.9676   | 0.4708 |
|        | WVD [87]        | 23.78  | 0.7948 | 0.1949  | 0.9935        | 0.9926   | 0.5795 | 15.72  | 0.5522 | 0.4510  | 0.9520      | 0.9597   | 0.5584 |
|        | Ours            | 27.05  | 0.8732 | 0.1352  | 0.9948        | 0.9946   | 0.6025 | 18.59  | 0.6149 | 0.3416  | 0.9685      | 0.9725   | 0.5623 |

Table 1. Quantitative Comparison of Appearance Generation. We compare both 1-view and 2-view based settings.

diffusion process, including a text prompt y, a condition image sequence Icond with a flexible number of available frames (where missing images are set to zero), corresponding binary masks M and optional per-view camera conditions Tcond. The overall diffusion process is defined as:

$$G_{\theta}: (\mathcal{Z}_t; t, \mathbf{y}, \mathcal{I}_{cond}, \mathcal{M}, \mathcal{T}_{cond}) \to \hat{\mathcal{Z}}_{t-1},$$
 (9)

where Z<sup>t</sup> is the noised latent at timestep t, Zˆ <sup>t</sup>−<sup>1</sup> is the predicted latent at t − 1.

Note that we do not provide geometric latents as condition signals, allowing the model to handle diverse tasks from input images only. During training, we uniformly sample conditions from the following options: (1) the first frame (1-view based), (2) the first and last frames (2-view based), (3) all frames, and adjust the binary masks correspondingly. We also randomly drop camera conditions to ensure they can be omitted during inference.

Inference. Practically, we evaluate on three conditioning settings: (1) 1-view-based generation, (2) 2-view-based generation, and (3) feed-forward reconstruction with a image sequence. Each setting can be performed with or without camera conditions. For fairness, we remove the camera conditions in the feed-forward reconstruction experiments.

## <span id="page-4-0"></span>3.3. Decoding Latents into Scene Attributes

Based on the pipeline described above, we achieve feedforward 3D scene generation by sampling unified latents from noise using Gθ, and decoding them into RGB frames and geometry attributes using *separate* decoders.

The appearance latents A ∈ R n×h×w×c are decoded by the pretrained RGB VAE D<sup>W</sup> to synthesize photorealistic video frames I. Similarly, the geometry latents G ∈ R n×h×w×c are mapped by the geometry adapter Dadp to recover geometry tokens V. These tokens are then decoded by pretrained VGGT heads D<sup>V</sup> to obtain scene attributes, including globally consistent point clouds P, perview depth maps D and camera parameters T . Following VGGT, we unproject the depth maps using the generated camera parameters as the final geometry results.

# 4. Experiments

In this section, we compare our method with state-of-theart approaches across various conditions. We first describe the training details in Sec. [4.1,](#page-4-1) followed by both quantitative and qualitative evaluations on 3D generation (Sec. [4.2\)](#page-5-0) and reconstruction (Sec. [4.3\)](#page-7-0). Finally, we present ablation studies to further validate the effectiveness of our approach in Sec. [4.4.](#page-7-1) We highlight the best , second-best , and third-best scores achieved on any metrics.

# <span id="page-4-1"></span>4.1. Training Details

Datasets. We train our model on a diverse collection of 3D datasets with camera calibrations, including: RealEstate10K [\[90\]](#page-11-13), DL3DV-10K [\[30\]](#page-9-23), ACID [\[31\]](#page-9-24), TartanAir [\[68\]](#page-10-22), KITTI-360 [\[26\]](#page-9-25), Waymo [\[54\]](#page-10-23), Co3Dv2 [\[43\]](#page-9-26), MVImgNet [\[83\]](#page-11-14), Virtual KITTI 2 [\[1\]](#page-8-14) and WildRGB-D [\[75\]](#page-10-24). Together, these datasets provide over 300k multi-view consistent 3D scenes, spanning a wide range of domains, including object-centric, indoor, outdoor, driving, and synthetic scenarios. For RealEstate10K, we follow the official train-test split, while for the other datasets, we randomly sample around 90% of the scenes for training and use the rest for testing. Text prompts for each scene are generated using a multi-modal large language model [\[2\]](#page-8-15). Notably, our method does not require dense reconstruction to obtain explicit 3D representations for training.

Implementation Details. For the geometry adapter, we adopt a causal autoencoder architecture similar to [\[62\]](#page-10-20), but with different input, output, and hidden dimensions. The adapter is trained on the mixed dataset described above. To ensure stability, the model is initially trained with 25 frames at a resolution of 560×560 for 15k iterations, using a batch size of 2 and gradient accumulation steps of 4 on 24 H20 GPUs, resulting a total batch size of 192. It is then finetuned with 49 frames for another 6k iterations, using a batch size of 1 and gradient accumulation steps of 8 on the same hardware. The adapter weights are randomly initialized.

For the video diffusion model, we fine-tune a pretrained

<span id="page-5-3"></span><span id="page-5-2"></span>

| ond.     | Method      |            | Co3Dv2         |        |            | WildRGB-D      |        | TartanAir  |                |        |  |
|----------|-------------|------------|----------------|--------|------------|----------------|--------|------------|----------------|--------|--|
| ပိ       |             | Accuracy ↓ | Completeness ↓ | CD↓    | Accuracy ↓ | Completeness ↓ | CD↓    | Accuracy ↓ | Completeness ↓ | CD↓    |  |
|          | Aether [59] | 1.2630     | 2.6366         | 1.9498 | 0.3181     | 0.2951         | 0.3066 | 3.1547     | 4.5366         | 3.8457 |  |
| view     | WVD [87]    | 1.8038     | 1.4237         | 1.6137 | 0.2708     | 0.2562         | 0.2635 | 4.3944     | 3.0660         | 3.7302 |  |
| <u>-</u> | VGGT [65]   | 0.3291     | 4.3830         | 2.3561 | 0.0346     | 0.6723         | 0.3534 | 0.7379     | 5.3595         | 3.0487 |  |
| _        | Ours        | 0.8284     | 1.3811         | 1.1047 | 0.1581     | 0.2402         | 0.1992 | 3.0250     | 2.5367         | 2.7809 |  |
|          | Aether [59] | 0.9664     | 2.1376         | 1.5520 | 0.3536     | 0.2540         | 0.3038 | 2.7745     | 3.4420         | 3.1082 |  |
| view     | WVD [87]    | 2.1153     | 1.3009         | 1.7081 | 0.2483     | 0.1813         | 0.2148 | 4.3794     | 2.5268         | 3.4531 |  |
| 2-vi     | VGGT [65]   | 0.3951     | 2.1566         | 1.2759 | 0.0276     | 0.2650         | 0.1463 | 0.9201     | 3.2554         | 2.0877 |  |
| ~        | Ours        | 0.7237     | 1.2298         | 0.9767 | 0.1109     | 0.1744         | 0.1426 | 2.2825     | 1.6643         | 1.9734 |  |

Table 2. Quantitative Comparison of Geometry Generation. We compare both 1-view and 2-view based settings.

<span id="page-5-1"></span>![](_page_5_Figure_2.jpeg)

Figure 4. **Qualitative Comparison of Novel View Synthesis** with 2-view conditions. The input images are shown on the left, and error maps are displayed overlaid on the results. Bluer colors indicate smaller errors, while redder colors indicate larger errors.

image-camera conditioned Wan2.1 [62]. Similar to the geometry adapter, during training we randomly sample 49 consecutive frames from each video clip, which are then resized and center-cropped to  $560 \times 560$ . The model is trained for 8k iterations with a batch size of 4 on 24 H20 GPUs. To enhance capability in handling diverse conditioning inputs, each training step has  $\frac{1}{3}$  probability of using (i) 1-view condition, (ii) 2-view (first-last frame) conditions, or (iii) the entire frame sequence as input. Additionally, the text prompt is dropped with a 20% probability for CFG [13], and the camera condition is omitted with a 50% probability.

#### <span id="page-5-0"></span>4.2. 3D Generation

**Datasets and Metrics.** We evaluate 3D generation on RealEstate10K [90], DL3DV-10K [30], Co3Dv2 [43], WildRGB-D [75] and TartanAir [68] datasets. For each task, we assess both appearance (RGB) and geometry (point clouds) metrics. Appearance metrics are computed across

all these datasets, while geometry metrics are evaluated only on Co3Dv2, WildRGB-D, and TartanAir, as the other datasets do not provide ground truth geometry. Note that we report appearance metrics only on RealEstate10K and DL3DV-10K, and geometry metrics only on Co3Dv2, WildRGB-D, and TartanAir in the main text. Please refer to Supp. for complete results on all datasets.

For appearance evaluation, we randomly sample 200 sequences with camera conditions from each of the RealEstate10K and DL3DV-10K, and compute PSNR, SSIM [70], and LPIPS [88] between the generated and ground-truth images. We additionally report the VBench Score [16, 17] to assess the models' generative capability, focusing on I2V Subject (I2V Subj.), I2V Background (I2V BG), and Imaging Quality (I.Q.) given the presence of image-based conditioning.

For geometry evaluation, we randomly sample 300 sequences with camera conditions from each of the Co3Dv2 and WildRGB-D, along with an additional 80 sequences

<span id="page-6-3"></span><span id="page-6-0"></span>

| Method          |            | Co3Dv2         |        |            | WildRGB-D      |        | TartanAir  |                |        |  |
|-----------------|------------|----------------|--------|------------|----------------|--------|------------|----------------|--------|--|
|                 | Accuracy ↓ | Completeness ↓ | CD ↓   | Accuracy ↓ | Completeness ↓ | CD ↓   | Accuracy ↓ | Completeness ↓ | CD ↓   |  |
| VGGT [65]       | 0.9157     | 1.0107         | 0.9632 | 0.0925     | 0.1405         | 0.1165 | 2.2929     | 0.8985         | 1.5957 |  |
| WVD (VAE only)  | 1.0533     | 1.3627         | 1.2080 | 0.1273     | 0.1780         | 0.1526 | 3.5337     | 2.0396         | 2.7867 |  |
| Ours (VAE only) | 0.9236     | 1.0735         | 0.9986 | 0.0929     | 0.1400         | 0.1165 | 2.2972     | 1.0063         | 1.6518 |  |
| Aether [59]     | 1.7755     | 1.2280         | 1.5018 | 0.3033     | 0.1665         | 0.2349 | 3.0287     | 2.3684         | 2.6985 |  |
| WVD [87]        | 1.7997     | 1.4609         | 1.6303 | 0.2758     | 0.1542         | 0.2150 | 3.8018     | 2.0820         | 2.9419 |  |
| Ours            | 0.9270     | 0.9980         | 0.9625 | 0.1058     | 0.1463         | 0.1260 | 1.9243     | 1.0959         | 1.5101 |  |

Table 3. Quantitative Comparison of Geometry Reconstruction. WVD (VAE only) uses pretrained RGB VAE to encode and reconstruct point clouds, while Ours (VAE only) projects and reconstruct VGGT tokens to decode scene geometry.

<span id="page-6-1"></span>![](_page_6_Picture_2.jpeg)

Figure 5. Quali. Comparison of Geometry Reconstruction.

from TartanAir. We first use the Umeyama algorithm [\[61\]](#page-10-26) to align the generated point clouds to the ground truth, then sample 20k points from both point clouds using Farthest Point Sampling (FPS) [\[41\]](#page-9-27), and finally compute Accuracy, Completeness, and Chamfer Distance (CD) [\[8\]](#page-8-19).

Comparison Baselines. We compare our method with several state-of-the-art approaches that use image and camera conditions, including *1) Reconstruction-based method:* DepthSplat [\[77\]](#page-11-12); *2) 2D generation methods:* LVSM [\[19\]](#page-8-13), Gen3C [\[44\]](#page-9-22), and Geometry Forcing (GF) [\[73\]](#page-10-21); and *3) Explicit 3D generation methods:* Aether [\[59\]](#page-10-12) and WVD [\[87\]](#page-11-3). We use the official implementations for all of these methods except for WVD, as it is not open-sourced; we re-implement it following the same training strategy as ours. Note that Aether does not output point maps, so we back-project its generated depths using predicted camera parameters to obtain point clouds.

Comparison on Appearance Generation. Tab. [1](#page-4-2) presents quantitative results for 1-view-based and 2-viewbased (first-last frames) appearance generation. Fig. [4](#page-5-1) provides the corresponding qualitative comparisons for the 2 view setting (see Supp. for 1-view results). Gen3R outperforms or matches the baselines in most cases.

- *1) Reconstruction-based methods:* We evaluate DepthSplat only in the 2-view setting, as it requires multi-view inputs to construct cost volumes. Although it achieves competitive results, it leaves holes in occluded regions (see Fig. [4\)](#page-5-1). In contrast, our method can plausibly complete these regions using diffusion-based generation.
- *2) 2D generation methods:* LVSM performs best in the 2 view case, as it is a non-generative model well suited for interpolation. However, it often produces blurred results in over-exposed scenes (Fig. [4,](#page-5-1) 1st row), and its performance degrades notably in the 1-view case. Gen3C also

<span id="page-6-2"></span>

| Cond.   |       | RealEstate10K |                                             |       | DL3DV-10K |               |        | Co3Dv2 WildRGB-D TartanAir |        |
|---------|-------|---------------|---------------------------------------------|-------|-----------|---------------|--------|----------------------------|--------|
| Method  |       |               | PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ |       |           |               | CD ↓   | CD ↓                       | CD ↓   |
| 1-view  |       |               |                                             |       |           |               |        |                            |        |
| 2-Stage | 17.38 |               | 0.6617 0.3412                               | 14.37 |           | 0.5085 0.5014 | 1.6223 | 0.2330                     | 3.7029 |
| w/o LKL | 16.31 |               | 0.6476 0.3941                               | 13.68 |           | 0.4797 0.5094 | 1.9620 | 0.3280                     | 4.0395 |
| Ours    | 20.51 |               | 0.7388 0.2281                               | 16.38 |           | 0.5821 0.4234 | 1.1047 | 0.1992                     | 2.7809 |
| 2-view  |       |               |                                             |       |           |               |        |                            |        |
| 2-Stage | 23.56 |               | 0.7883 0.1931                               | 15.92 |           | 0.5413 0.4427 | 1.3615 | 0.1623                     | 2.6579 |
| w/o LKL | 21.62 |               | 0.7592 0.2185                               | 15.41 |           | 0.5222 0.4527 | 1.7144 | 0.2898                     | 3.5508 |
| Ours    | 27.05 |               | 0.8732 0.1352                               | 18.59 |           | 0.6149 0.3416 | 0.9767 | 0.1426                     | 1.9734 |

Table 4. Ablation Study on appearance and geometry generation.

achieves competitive results in 1-view generation by combining depth-based warping and inpainting, but its quality heavily depends on depth accuracy, leading to misaligned boundaries when the depth estimates are inaccurate. In addition, it sometimes exhibits color differences from the input image, as shown in Fig. [4.](#page-5-1) More relevant to our method, GF similarly attempts to bridge reconstruction and generation. Unlike ours, which aligns latent spaces *prior* to diffusion training, GF aligns intermediate diffusion features to the reconstruction model *during* training, which is less effective in practice. Finally, all of the above baselines operate purely in 2D and do not produce any 3D outputs.

*3) Explicit 3D generation methods:* Our method clearly surpasses the most relevant generative baselines, Aether and WVD, both of which jointly generate RGB images and scene geometry. As shown in Fig. [4,](#page-5-1) our approach yields higher-quality results and better camera alignment than WVD. This highlights the advantage of bridging reconstruction and generation models in the latent space, rather than compressing the reconstruction outputs for generation.

Comparison on Geometry Generation. We further compare the generated point clouds with 3D-based methods in Tab. [2,](#page-5-2) and Fig. [3](#page-3-1) visualizes point clouds generated from a single input view. Our method clearly outperforms Aether and WVD in CD across both generation settings. The qualitative results in Fig. [3](#page-3-1) are consistent with the quantitative findings: Aether and WVD exhibit poor global consistency, whereas our method produces more complete objects and scenes from single-view observations, with plausible geometry in unseen regions. We also include VGGT in Tab. [2](#page-5-2) as a reference. It performs pure reconstruction from one or two input views without generation. Although VGGT achieves better accuracy, it suffers from lower completeness since it does not generate geometry for novel views, leading to a

<span id="page-7-5"></span><span id="page-7-3"></span>![](_page_7_Figure_0.jpeg)

Figure 6. **Appearance Comparison** with ablation baselines. We highlight the artifacts of the baselines directly in the figures.

<span id="page-7-2"></span>

| Method                          | Cond. | RealEstate10K | WildRGB-D | nd. | RealEstate10K | WildRGB-D |
|---------------------------------|-------|---------------|-----------|-----|---------------|-----------|
| meurea                          | වී    | AUC@30↑       | AUC@30↑   | ్త్ | AUC@30↑       | AUC@30↑   |
| Aether [59]                     |       | 0.6398        | 0.5375    |     | 0.6220        | 0.5068    |
| WVD [87]                        | ≽     | 0.6727 0.6780 |           | 3   | 0.7249        | 0.7113    |
| 2-Stage                         | į.    | 0.6832        | 0.6798    | ķ.  | 0.7188        | 0.7211    |
| w/o $\mathcal{L}_{\mathrm{KL}}$ | -     | 0.4100        | 0.4759    | 4   | 0.4683        | 0.4947    |
| Ours                            |       | 0.7443        | 0.8004    |     | 0.7732        | 0.8098    |

Table 5. Quantitative Comparison of Camera Controllability on RealEstate10K and WildRGB-D datasets.

worse CD than Gen3R.

#### <span id="page-7-0"></span>4.3. Feed-forward 3D Reconstruction

Dataset and Metrics. Similar to 3D generation, we evaluate feed-forward 3D reconstruction on the same sequences sample from Co3Dv2, WildRGB-D and TartanAir datasets as before, but *without* camera conditions. We assess both geometric quality and camera pose estimation. For geometry, we first align the predicted point clouds with GT, downsample both using FPS, and then compute Accuracy, Completeness, and Chamfer Distance (CD). For camera pose estimation, we use the RealEstate10K and WildRGB-D datasets and follow VGGT [65] in reporting AUC@30, which combines both Relative Rotation Accuracy (RRA) and Relative Translation Accuracy (RTA). Please refer to Supplementary for camera pose estimation results.

**Comparison Baselines.** We compare our method with 1) feed-forward 3D reconstruction approach, VGGT, as well as different variants of VAE for compressing VGGT. WVD (VAE only) encodes and decodes VGGT's global point clouds using a pre-trained RGB VAE, while Ours (VAE only) encodes VGGT's geometry tokens through our adapter and decodes them back. We also compare against 2) 3D generation methods Aether and WVD.

**Comparison on Geometry Reconstruction.** We present quantitative results in Tab. 3. 1) Feed-forward 3D reconstruction: Our VAE maintains the competitive performance of VGGT, whereas WVD's VAE produces subpar results when encoding and reconstructing explicit point clouds, as it is originally designed for RGB image reconstruction. Moreover, our generative version even enhances the reconstruction performance. This improvement arises because our method jointly models the appearance and geom-

<span id="page-7-4"></span>![](_page_7_Picture_9.jpeg)

Figure 7. Visualization of Latent Spaces from different VAEs.

etry distribution, enabling mutual interactions between the two modalities and thereby refining noisy geometric predictions. As shown in Fig. 5, VGGT occasionally exhibits floaters in its predicted geometry, and our adapted VAE inherits these artifacts. However, our generative model corrects the errors and produces cleaner depth. 2) 3D generation methods: Our method significantly outperforms existing generative models, Aether and WVD, on the reconstruction task, despite that our re-implemented WVD also leverages the prior of VGGT.

### <span id="page-7-1"></span>4.4. Ablation Study

Effect of Joint Generation. We investigate the impact of jointly generating RGB and geometry. To this end, we design a 2-Stage baseline: a video diffusion model generates only RGB under camera control, while geometry is predicted separately using VGGT from the generated images. Results in Tab. 4 show that our joint generation approach outperforms the 2-Stage pipeline in both appearance and geometry. This is because the 2-Stage approach naively connects 2D generation with 3D reconstruction, leading to accumulated errors. Besides, Tab. 5 and Fig. 6 further show that our method outperforms this 2-Stage alternative in terms of camera control accuracy.

**Effect of the Distribution Alignment Loss.** We further evaluate the impact of our distribution alignment loss  $\mathcal{L}_{\mathrm{KL}}$  by training a variant of the adapter without it and visualizing the resulting latents in Fig. 7. Without this constraint, the geometry latents clearly deviate from the appearance latents. Results in Tab. 4, Tab. 5, and Fig. 6 show that this misalignment hinders convergence and significantly degrades both camera controllability and generation quality.

#### 5. Conclusion

We introduced Gen3R, a unified framework that couples feed-forward reconstruction with video diffusion for high-fidelity 3D scene synthesis. By reformulating VGGT as an asymmetric geometry VAE and aligning its latents with a video diffusion model, Gen3R jointly generates RGB videos and globally consistent 3D geometry. Extensive experiments show that Gen3R outperforms existing 2D and 3D based generative methods in both appearance and geometry, while also delivering superior camera controllability. Furthermore, Gen3R improves the robustness of feed-forward reconstruction, highlighting

the mutual benefits of combining generative priors with strong geometric foundations. We believe Gen3R offers a promising direction toward controllable and high-fidelity 3D scene generation, and opens new possibilities for bridging reconstruction and generative modeling at scale.

# References

- <span id="page-8-14"></span>[1] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2, 2020. [5](#page-4-3)
- <span id="page-8-15"></span>[2] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruiliang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. Internlm2 technical report, 2024. [5](#page-4-3)
- <span id="page-8-7"></span>[3] Carlos Campos, Richard Elvira, Juan J Gomez Rodr ´ ´ıguez, Jose MM Montiel, and Juan D Tard ´ os. Orb-slam3: An ´ accurate open-source library for visual, visual–inertial, and multimap slam. *IEEE transactions on robotics*, 37(6):1874– 1890, 2021. [3](#page-2-2)
- <span id="page-8-5"></span>[4] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. Mvsplat360: Feed-forward 360 scene synthesis from sparse views. *Advances in Neural Information Processing Systems*, 37:107064–107086, 2024. [2](#page-1-0)
- <span id="page-8-12"></span>[5] Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng Tao, He Zhu, Fangzhou Hong, Liang Pan, and Ziwei Liu. 4dnex: Feed-forward 4d generative modeling made easy. *arXiv preprint arXiv:2508.13154*, 2025. [4](#page-3-2)
- <span id="page-8-0"></span>[6] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. *arXiv preprint arXiv:2311.13384*, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-6"></span>[7] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects, 2022. [2](#page-1-0)
- <span id="page-8-19"></span>[8] Haoqiang Fan, Hao Su, and Leonidas Guibas. A point set

- generation network for 3d object reconstruction from a single image, 2016. [7](#page-6-3)
- <span id="page-8-1"></span>[9] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. Scenescape: Text-driven consistent scene generation. *Advances in Neural Information Processing Systems*, 36:39897–39914, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-2"></span>[10] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. *arXiv preprint arXiv:2405.10314*, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-4"></span>[11] Hyojun Go, Byeongjun Park, Jiho Jang, Jin-Young Kim, Soonwoo Kwon, and Changick Kim. Splatflow: Multi-view rectified flow model for 3d gaussian splatting synthesis. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 21524–21536, 2025. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-3"></span>[12] Junlin Hao, Peiheng Wang, Haoyang Wang, Xinggong Zhang, and Zongming Guo. Gaussvideodreamer: 3d scene generation with video diffusion and inconsistency-aware gaussian splatting. *arXiv preprint arXiv:2504.10001*, 2025. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-16"></span>[13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. [6](#page-5-3)
- <span id="page-8-11"></span>[14] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. *arXiv preprint arXiv:2205.15868*, 2022. [3](#page-2-2)
- <span id="page-8-8"></span>[15] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d, 2024. [3](#page-2-2)
- <span id="page-8-17"></span>[16] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024. [6,](#page-5-3) [2](#page-1-0)
- <span id="page-8-18"></span>[17] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench++: Comprehensive and versatile benchmark suite for video generative models. *arXiv preprint arXiv:2411.13503*, 2024. [6,](#page-5-3) [2](#page-1-0)
- <span id="page-8-10"></span>[18] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. *arXiv preprint arXiv:2505.23716*, 2025. [3](#page-2-2)
- <span id="page-8-13"></span>[19] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias, 2025. [5,](#page-4-3) [6,](#page-5-3) [7,](#page-6-3) [2,](#page-1-0) [3](#page-2-2)
- <span id="page-8-9"></span>[20] Nikhil Keetha, Norman Muller, Johannes Sch ¨ onberger, ¨ Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes,

- Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulo, Christian Richardt, Deva Ramanan, Sebastian Scherer, ` and Peter Kontschieder. Mapanything: Universal feedforward metric 3d reconstruction, 2025. [3](#page-2-2)
- <span id="page-9-7"></span>[21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, ¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42 (4), 2023. [2](#page-1-0)
- <span id="page-9-16"></span>[22] Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud. Ground- ˆ ing image matching in 3d with mast3r, 2024. [2,](#page-1-0) [3](#page-2-2)
- <span id="page-9-19"></span>[23] Hao Li, Zhengyu Zou, Fangfu Liu, Xuanyang Zhang, Fangzhou Hong, Yukang Cao, Yushi Lan, Manyuan Zhang, Gang Yu, Dingwen Zhang, and Ziwei Liu. Iggt: Instancegrounded geometry transformer for semantic 3d reconstruction, 2025. [3](#page-2-2)
- <span id="page-9-4"></span>[24] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Director3d: Real-world camera trajectory and 3d scene generation from text. *Advances in neural information processing systems*, 37:75125–75151, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-13"></span>[25] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N. Plataniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Navigating 3d scenes from a single image, 2025. [2](#page-1-0)
- <span id="page-9-25"></span>[26] Yiyi Liao, Jun Xie, and Andreas Geiger. Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d, 2022. [5](#page-4-3)
- <span id="page-9-12"></span>[27] Chendi Lin, Heshan Liu, Qunshu Lin, Zachary Bright, Shitao Tang, Yihui He, Minghao Liu, Ling Zhu, and Cindy Le. Objaverse++: Curated 3d object dataset with quality annotations, 2025. [2](#page-1-0)
- <span id="page-9-9"></span>[28] Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li, and Yadong Mu. Diffsplat: Repurposing image diffusion models for scalable gaussian splat generation, 2025. [2](#page-1-0)
- <span id="page-9-0"></span>[29] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 300–309, 2022. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-23"></span>[30] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi Zhang, Bedrich Benes, and Aniket Bera. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision, 2023. [5,](#page-4-3) [6,](#page-5-3) [2](#page-1-0)
- <span id="page-9-24"></span>[31] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 14458–14467, 2021. [5](#page-4-3)
- <span id="page-9-8"></span>[32] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model, 2024. [2](#page-1-0)

- <span id="page-9-2"></span>[33] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. *arXiv preprint arXiv:2309.03453*, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-20"></span>[34] Yang Liu, Chuanchen Luo, Zimo Tang, Junran Peng, and Zhaoxiang Zhang. Vggt-x: When vggt meets dense novel view synthesis, 2025. [3](#page-2-2)
- <span id="page-9-10"></span>[35] Xuyi Meng, Chen Wang, Jiahui Lei, Kostas Daniilidis, Jiatao Gu, and Lingjie Liu. Zero-1-to-g: Taming pretrained 2d diffusion model for direct 3d generation, 2025. [2](#page-1-0)
- <span id="page-9-6"></span>[36] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf. *Communications of the ACM*, 65:99 – 106, 2020. [2](#page-1-0)
- <span id="page-9-17"></span>[37] Raul Mur-Artal, J. M. M. Montiel, and Juan D. Tardos. Orbslam: A versatile and accurate monocular slam system. *IEEE Transactions on Robotics*, 31(5):1147–1163, 2015. [3](#page-2-2)
- <span id="page-9-14"></span>[38] Norman Muller, Yawar Siddiqui, Lorenzo Porzi, ¨ Samuel Rota Bulo, Peter Kontschieder, and Matthias ` Nießner. Diffrf: Rendering-guided 3d radiance field diffusion, 2023. [2](#page-1-0)
- <span id="page-9-15"></span>[39] Hieu T. Nguyen, Yiwen Chen, Vikram Voleti, Varun Jampani, and Huaizu Jiang. Housecrafter: Lifting floorplans to 3d scenes with 2d diffusion model, 2025. [2](#page-1-0)
- <span id="page-9-1"></span>[40] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. *arXiv preprint arXiv:2209.14988*, 2022. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-27"></span>[41] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space, 2017. [7](#page-6-3)
- <span id="page-9-21"></span>[42] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi- ´ sion transformers for dense prediction, 2021. [3](#page-2-2)
- <span id="page-9-26"></span>[43] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In *International Conference on Computer Vision*, 2021. [5,](#page-4-3) [6,](#page-5-3) [2](#page-1-0)
- <span id="page-9-22"></span>[44] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Muller, Alexan- ¨ der Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control, 2025. [5,](#page-4-3) [6,](#page-5-3) [7,](#page-6-3) [2,](#page-1-0) [3](#page-2-2)
- <span id="page-9-11"></span>[45] Barbara Roessle, Norman Muller, Lorenzo Porzi, Samuel ¨ Rota Bulo, Peter Kontschieder, Angela Dai, and Matthias ` Nießner. L3dg: Latent 3d gaussian diffusion. In *SIGGRAPH Asia 2024 Conference Papers*, page 1–11. ACM, 2024. [2](#page-1-0)
- <span id="page-9-5"></span>[46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn- ¨ thesis with latent diffusion models, 2021. [2](#page-1-0)
- <span id="page-9-3"></span>[47] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360 degree view synthesis from a single real image. 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-18"></span>[48] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4104–4113, 2016. [3](#page-2-2)

- <span id="page-10-5"></span>[49] Katja Schwarz, Seung Wook Kim, Jun Gao, Sanja Fidler, Andreas Geiger, and Karsten Kreis. Wildfusion: Learning 3d-aware latent diffusion models in view space. In *International Conference on Learning Representations (ICLR)*, 2024. [2](#page-1-0)
- <span id="page-10-10"></span>[50] Katja Schwarz, Norman Mueller, and Peter Kontschieder. Generative gaussian splatting: Generating 3d scenes with video diffusion priors, 2025. [2](#page-1-0)
- <span id="page-10-9"></span>[51] Katja Schwarz, Denys Rozumnyi, Samuel Rota Bulo,` Lorenzo Porzi, and Peter Kontschieder. A recipe for generating 3d worlds from a single image, 2025. [2](#page-1-0)
- <span id="page-10-18"></span>[52] You Shen, Zhipeng Zhang, Yansong Qu, and Liujuan Cao. Fastvggt: Training-free acceleration of visual geometry transformer. 2025. [3](#page-2-2)
- <span id="page-10-2"></span>[53] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-23"></span>[54] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Sheng Zhao, Shuyang Cheng, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset, 2020. [5](#page-4-3)
- <span id="page-10-8"></span>[55] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. In *International Conference on Computer Vision (ICCV)*, 2025. [2](#page-1-0)
- <span id="page-10-4"></span>[56] Stanislaw Szymanowicz, Jason Y. Zhang, Pratul Srinivasan, Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo Martin-Brualla, Jonathan T. Barron, and Philipp Henzler. Bolt3d: Generating 3d scenes in seconds, 2025. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-0"></span>[57] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. *arXiv preprint arXiv:2309.16653*, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-14"></span>[58] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. *arXiv preprint arXiv:2402.05054*, 2024. [3](#page-2-2)
- <span id="page-10-12"></span>[59] Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, and Tong He. Aether: Geometric-aware unified world modeling. *arXiv preprint arXiv:2503.18945*, 2025. [2,](#page-1-0) [4,](#page-3-2) [5,](#page-4-3) [6,](#page-5-3) [7,](#page-6-3) [8,](#page-7-5) [1,](#page-0-0) [3](#page-2-2)
- <span id="page-10-13"></span>[60] Zachary Teed and Jia Deng. DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras. *Advances in neural information processing systems*, 2021. [3](#page-2-2)
- <span id="page-10-26"></span>[61] S. Umeyama. Least-squares estimation of transformation parameters between two point patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 13(4):376–380, 1991. [7](#page-6-3)
- <span id="page-10-20"></span>[62] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video gen-

- erative models. *arXiv preprint arXiv:2503.20314*, 2025. [3,](#page-2-2) [4,](#page-3-2) [5,](#page-4-3) [6,](#page-5-3) [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-17"></span>[63] Chaoyang Wang, Ashkan Mirzaei, Vidit Goel, Willi Menapace, Aliaksandr Siarohin, Avalon Vinella, Michael Vasilkovsky, Ivan Skorokhodov, Vladislav Shakhrai, Sergey Korolev, Sergey Tulyakov, and Peter Wonka. 4real-videov2: Fused view-time attention and feedforward reconstruction for 4d scene generation. *ArXiv*, abs/2506.18839, 2025. [3](#page-2-2)
- <span id="page-10-15"></span>[64] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 21686–21697, 2024. [3](#page-2-2)
- <span id="page-10-7"></span>[65] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 5294–5306, 2025. [2,](#page-1-0) [3,](#page-2-2) [6,](#page-5-3) [7,](#page-6-3) [8,](#page-7-5) [1](#page-0-0)
- <span id="page-10-16"></span>[66] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state, 2025. [3](#page-2-2)
- <span id="page-10-6"></span>[67] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy, 2024. [2,](#page-1-0) [3](#page-2-2)
- <span id="page-10-22"></span>[68] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. 2020. [5,](#page-4-3) [6,](#page-5-3) [2](#page-1-0)
- <span id="page-10-19"></span>[69] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π 3 : Scalable permutation-equivariant visual geometry learning, 2025. [3](#page-2-2)
- <span id="page-10-25"></span>[70] Zhou Wang, Alan Conrad Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13:600–612, 2004. [6](#page-5-3)
- <span id="page-10-1"></span>[71] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. *Advances in neural information processing systems*, 36: 8406–8441, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-11"></span>[72] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and Jan Eric Lenssen. latentsplat: Autoencoding variational gaussians for fast generalizable 3d reconstruction, 2024. [2](#page-1-0)
- <span id="page-10-21"></span>[73] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling, 2025. [5,](#page-4-3) [7,](#page-6-3) [2](#page-1-0)
- <span id="page-10-3"></span>[74] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 21551–21561, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-24"></span>[75] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos, 2024. [5,](#page-4-3) [6,](#page-5-3) [2](#page-1-0)

- <span id="page-11-5"></span>[76] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation, 2025. [2](#page-1-0)
- <span id="page-11-12"></span>[77] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth, 2025. [5,](#page-4-3) [6,](#page-5-3) [7,](#page-6-3) [2](#page-1-0)
- <span id="page-11-11"></span>[78] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. *arXiv:2406.09414*, 2024. [3](#page-2-2)
- <span id="page-11-2"></span>[79] Yuanbo Yang, Jiahao Shao, Xinyang Li, Yujun Shen, Andreas Geiger, and Yiyi Liao. Prometheus: 3d-aware latent diffusion models for feed-forward text-to-3d scene generation, 2025. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-11-10"></span>[80] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. *arXiv preprint arXiv:2408.06072*, 2024. [3](#page-2-2)
- <span id="page-11-0"></span>[81] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjourney: Going from anywhere to everywhere. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6658–6667, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-11-1"></span>[82] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 5916–5926, 2025. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-11-14"></span>[83] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. Mvimgnet: A large-scale dataset of multi-view images. In *CVPR*, 2023. [5](#page-4-3)
- <span id="page-11-7"></span>[84] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion, 2025. [3](#page-2-2)
- <span id="page-11-8"></span>[85] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting, 2024. [3](#page-2-2)
- <span id="page-11-6"></span>[86] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets, 2024. [2](#page-1-0)
- <span id="page-11-3"></span>[87] Qihang Zhang, Shuangfei Zhai, Miguel Angel Bautista Martin, Kevin Miao, Alexander Toshev, Joshua Susskind, and Jiatao Gu. World-consistent video diffusion with explicit 3d modeling. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 21685–21695, 2025. [1,](#page-0-0) [2,](#page-1-0) [4,](#page-3-2) [5,](#page-4-3) [6,](#page-5-3) [7,](#page-6-3) [8,](#page-7-5) [3](#page-2-2)
- <span id="page-11-15"></span>[88] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric, 2018. [6](#page-5-3)
- <span id="page-11-4"></span>[89] Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan, Linjie Li, Zhengyuan Yang, Jianfeng Wang, Gim Hee Lee,

- and Lijuan Wang. Genxd: Generating any 3d and 4d scenes. In *ICLR*, 2025. [2](#page-1-0)
- <span id="page-11-13"></span>[90] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images, 2018. [5,](#page-4-3) [6,](#page-5-3) [2](#page-1-0)
- <span id="page-11-9"></span>[91] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. *arXiv preprint arXiv:2507.11539*, 2025. [3](#page-2-2)

# Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction

# Supplementary Material

## 6. Implementation Details

### **6.1. Processing Input Conditions**

We employ multiple conditions into the diffusion process, including a text prompt  $\mathbf{y}$ , a condition image sequence  $\mathcal{I}_{cond}$  with a flexible number of available frames (missing images are set to zero), corresponding binary masks  $\mathcal{M}$  and optional per-view camera conditions  $\mathcal{T}_{cond}$ . The condition images  $\mathcal{I}_{cond}$  are encoded into appearance latents  $\mathcal{A}_{cond}$  by pretrained RGB VAE  $\mathcal{E}_{\mathcal{W}}$ :

$$\mathcal{E}_{\mathcal{W}}: \mathcal{I}_{cond} \to \mathcal{A}_{cond} \in \mathbb{R}^{n \times h \times w \times c},$$
 (10)

while the masks  $\mathcal{M}$  are downsampled to  $\mathcal{M}_a \in \mathbb{R}^{n \times h \times w \times 4}$  to match the latent resolution. To ensure dimensional consistency with the noised latents, we initialize the geometry branch's condition latents  $\mathcal{G}_{cond} \in \mathbb{R}^{n \times h \times w \times c}$  and corresponding masks  $\mathcal{M}_g \in \mathbb{R}^{n \times h \times w \times 4}$  as **zeros**.

Finally, the appearance and geometry latents are fused with their respective latent masks along the channel dimension, and the two modalities are further concatenated in the width dimension to construct the unified condition latent:

$$\mathcal{Z}_{cond} = [\mathcal{A}_{cond} \oplus \mathcal{M}_a; \mathcal{G}_{cond} \oplus \mathcal{M}_g] \in \mathbb{R}^{n \times h \times 2w \times c'}, \tag{11}$$
where  $(\cdot, \oplus \cdot)$  means concatenation along channel dimension

where  $(\cdot \oplus \cdot)$  means concatenation along channel dimension, and c' = c + 4. The input to the diffusion model is then constructed by concatenating the noised latents  $\mathcal{Z}_t$  with the condition latents  $\mathcal{Z}_{cond}$  along the channel dimension:

$$\mathcal{Z}_{in} = \mathcal{Z}_t \oplus \mathcal{Z}_{cond},$$
 (12)

$$G_{\theta}: \mathcal{Z}_{in} \to \hat{\mathcal{Z}}_{t-1}.$$
 (13)

### **6.2. Model Architectures**

**Geometry Adapter.** We obtain our adapter  $(\mathcal{E}_{adp}, \mathcal{D}_{adp})$  by modifying Wan's causal VAE [62]. The adapter projects VGGT [65] geometry tokens  $\mathcal{V} \in \mathbb{R}^{N \times L \times h_v \times w_v \times C}$  into the video diffusion model's latent space and maps them back:

$$\mathcal{E}_{\text{adp}}: \mathcal{V} \to \mathcal{G} \in \mathbb{R}^{n \times h \times w \times c},$$
 (14)

$$\mathcal{D}_{\text{adp}}: \mathcal{G} \to \mathcal{V} \in \mathbb{R}^{N \times L \times h_v \times w_v \times C}, \tag{15}$$

where L=5, since we broadcast VGGT's camera tokens of each frame to the spatial resolution  $h_v \times w_v$  ( $h_v=w_v=40$ ), and concatenate it with the other 4 tokens along the L dimension

To match the VAE input format, we first reshape  $\mathcal{V}$  into  $\mathcal{V}' \in \mathbb{R}^{N \times h_v \times w_v \times (L \times C)}$ . Accordingly, we set the adapter input dimension to  $L \times C = 10240$  and use hidden dimensions [512, 256, 128, 128]. We then re-sample the input tokens  $\mathcal{V}'$  to a spatial resolution of  $h \times w = 70 \times 70$ 

<span id="page-12-0"></span>![](_page_12_Figure_18.jpeg)

Figure 8. Qualitative Comparison of Geometry Generation in the 2-view based setting.

<span id="page-12-1"></span>![](_page_12_Figure_20.jpeg)

Figure 9. Qualitative Comparison of Geometry Reconstruction.

using nearest-exact interpolation, and apply a 2D convolution to project the channels to 1024. The resulting features are processed by causal convolution layers, where we keep the spatial resolution unchanged, yielding geometry latents  $\mathcal{G} \in \mathbb{R}^{n \times h \times w \times c}$ . Similarly, the decoder  $\mathcal{D}_{adp}$  mirrors the encoder architecture in reverse, reconstructing the geometry tokens  $\mathcal{V}$  from the latents  $\mathcal{G}$ .

**Diffusion Transformer.** We adapt the DiT architecture from VideoX-Fun's Wan2.1 [62] to accommodate our joint appearance-geometry latents. Specifically, we set the input channel dimension to  $c+c^\prime=36$ . To support width-wise concatenation of appearance and geometry latents, we modify the positional embeddings so that corresponding pixels in the left and right halves of the latents share identical RoPE embeddings.

<span id="page-13-0"></span>

| Cond.           |        |        | C       | o3Dv2      |         |        |         |        | Wild    | iRGB-D     |         |        | TartanAir |        |         |            |         |        |
|-----------------|--------|--------|---------|------------|---------|--------|---------|--------|---------|------------|---------|--------|-----------|--------|---------|------------|---------|--------|
| Method          | PSNR ↑ | SSIM ↑ | LPIPS ↓ | I2V Subj.↑ | I2V BG↑ | I.Q. ↑ | PSNR ↑  | SSIM ↑ | LPIPS ↓ | I2V Subj.↑ | I2V BG↑ | I.Q. ↑ | PSNR ↑    | SSIM ↑ | LPIPS ↓ | I2V Subj.↑ | I2V BG↑ | I.Q.↑  |
| 1-view          |        |        |         |            |         |        |         |        |         |            |         |        |           |        |         |            |         |        |
| LVSM [19]       | 14.08  | 0.5623 | 0.5698  | 0.9482     | 0.9581  | 0.3579 | 13.9483 | 0.5239 | 0.5195  | 0.9692     | 0.9713  | 0.4004 | 14.44     | 0.5044 | 0.5210  | 0.9325     | 0.9540  | 0.3542 |
| Gen3C [44]      | 15.82  | 0.5666 | 0.5095  | 0.9134     | 0.9355  | 0.4335 | 14.60   | 0.5463 | 0.4513  | 0.9622     | 0.9646  | 0.4629 | 13.95     | 0.4731 | 0.5385  | 0.9142     | 0.9403  | 0.3713 |
| GF [73]         | 10.25  | 0.3150 | 0.6761  | 0.7933     | 0.8193  | 0.5320 | 11.8944 | 0.4147 | 0.5940  | 0.9215     | 0.9214  | 0.5310 | 10.21     | 0.3249 | 0.6249  | 0.7447     | 0.7864  | 0.4379 |
| Aether [59]     | 12.78  | 0.5106 | 0.6052  | 0.9229     | 0.9395  | 0.4411 | 11.87   | 0.4289 | 0.5973  | 0.9595     | 0.9614  | 0.4786 | 12.88     | 0.4585 | 0.5645  | 0.9295     | 0.9480  | 0.4303 |
| WVD [87]        | 13.35  | 0.4733 | 0.5765  | 0.9339     | 0.9484  | 0.5355 | 12.95   | 0.4522 | 0.5362  | 0.9669     | 0.9671  | 0.5513 | 12.77     | 0.4513 | 0.5652  | 0.9271     | 0.9473  | 0.4571 |
| Ours            | 16.09  | 0.5754 | 0.4997  | 0.9535     | 0.9588  | 0.5383 | 14.73   | 0.5501 | 0.4398  | 0.9715     | 0.9716  | 0.5609 | 15.04     | 0.5069 | 0.5073  | 0.9350     | 0.9546  | 0.4620 |
| 2-view          |        |        |         |            |         |        |         |        |         |            |         |        |           |        |         |            |         |        |
| DepthSplat [77] | 10.45  | 0.3262 | 0.6167  | 0.8314     | 0.8585  | 0.2992 | 16.22   | 0.5382 | 0.4518  | 0.9012     | 0.9067  | 0.3779 | 13.87     | 0.4585 | 0.5195  | 0.8073     | 0.8474  | 0.3301 |
| LVSM [19]       | 17.87  | 0.5986 | 0.4534  | 0.9467     | 0.9519  | 0.4064 | 19.13   | 0.6789 | 0.3134  | 0.9747     | 0.9730  | 0.4555 | 17.79     | 0.5685 | 0.4265  | 0.9415     | 0.9569  | 0.3628 |
| Gen3C [44]      | 17.16  | 0.5927 | 0.4776  | 0.9149     | 0.9361  | 0.4263 | 17.81   | 0.6307 | 0.3882  | 0.9636     | 0.9651  | 0.4634 | 15.24     | 0.5055 | 0.5318  | 0.9119     | 0.9376  | 0.3668 |
| GF [73]         | 12.67  | 0.3855 | 0.5998  | 0.7645     | 0.7925  | 0.4969 | 13.51   | 0.3991 | 0.4609  | 0.8785     | 0.8852  | 0.5374 | 12.06     | 0.3670 | 0.5666  | 0.7447     | 0.7946  | 0.4589 |
| Aether [59]     | 14.28  | 0.5405 | 0.5498  | 0.9322     | 0.9426  | 0.4647 | 13.79   | 0.4884 | 0.5161  | 0.9491     | 0.9512  | 0.4685 | 14.53     | 0.4989 | 0.5153  | 0.9294     | 0.9496  | 0.4267 |
| WVD [87]        | 14.66  | 0.5101 | 0.5334  | 0.9246     | 0.9409  | 0.5306 | 16.27   | 0.5421 | 0.4098  | 0.9631     | 0.9646  | 0.5627 | 14.22     | 0.4605 | 0.5266  | 0.9116     | 0.9371  | 0.4680 |
| Ours            | 18.01  | 0.6085 | 0.4371  | 0.9547     | 0.9597  | 0.5405 | 18.88   | 0.6448 | 0.3256  | 0.9746     | 0.9755  | 0.5685 | 17.34     | 0.5581 | 0.4416  | 0.9385     | 0.9559  | 0.4748 |

Table 6. **Quantitative Comparison of Appearance Generation.** We compare both 1-view and 2-view settings with camera conditions.

<span id="page-13-1"></span>

| ond.   | Method      |               | Reall         | Estate10K     | DL3DV-10K     |               |               |               |               |               |               |
|--------|-------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| ပိ     |             | I2V Subj.↑    | I2V BG↑       | Aes.Q.↑       | I.Q. ↑        | M.S.↑         | I2V Subj. ↑   | I2V BG↑       | Aes.Q.↑       | I.Q.↑         | M.S.↑         |
| 1-view | Aether [59] | 0.9743        | 0.9770        | 0.5118        | 0.5060        | 0.9885        | 0.9377        | 0.9501        | 0.4704        | 0.4872        | 0.9600        |
|        | WVD [87]    | 0.9815        | 0.9843        | 0.5125        | 0.5653        | 0.9895        | 0.9274        | 0.9412        | 0.4555        | 0.4916        | 0.9542        |
|        | Ours        | <b>0.9879</b> | <b>0.9890</b> | <b>0.5291</b> | <b>0.5761</b> | <b>0.9929</b> | <b>0.9461</b> | <b>0.9561</b> | <b>0.4727</b> | <b>0.5187</b> | <b>0.9701</b> |
| 2-view | Aether [59] | 0.9852        | 0.9843        | 0.5278        | 0.5187        | 0.9923        | 0.9485        | 0.9521        | 0.4846        | 0.5026        | 0.9685        |
|        | WVD [87]    | 0.9929        | 0.9923        | 0.5336        | 0.5973        | 0.9938        | 0.9403        | 0.9518        | 0.4760        | 0.5338        | 0.9685        |
|        | Ours        | <b>0.9949</b> | <b>0.9947</b> | <b>0.5369</b> | <b>0.6009</b> | <b>0.9947</b> | <b>0.9549</b> | <b>0.9576</b> | <b>0.4881</b> | <b>0.5357</b> | <b>0.9719</b> |

Table 7. Quantitative Comparison of Appearance Generation without camera conditions.

<span id="page-13-2"></span>

| Method | RealEstate10K | WildRGB-D |  |  |  |
|--------|---------------|-----------|--|--|--|
|        | AUC@30↑       | AUC@30↑   |  |  |  |
| Aether | 0.7291        | 0.7303    |  |  |  |
| VGGT   | 0.8387        | 0.8406    |  |  |  |
| Ours   | 0.8265        | 0.8391    |  |  |  |

Table 8. Quantitative Comparison of Camera Pose Estimation in feed-forward 3D reconstruction.

<span id="page-13-3"></span>

| Method       | R       | ealEstate10 | )K      | DL3DV-10K |        |         |  |  |  |
|--------------|---------|-------------|---------|-----------|--------|---------|--|--|--|
| Memod        | PSNR ↑  | SSIM ↑      | LPIPS ↓ | PSNR ↑    | SSIM ↑ | LPIPS ↓ |  |  |  |
| VGGT* [65]   | 23.3927 | 0.8346      | 0.2341  | 22.6958   | 0.7557 | 0.2910  |  |  |  |
| RGB VAE [62] | 37.5770 | 0.9819      | 0.0288  | 32.7673   | 0.9057 | 0.1031  |  |  |  |

Table 9. **Quantitative Comparison for RGB Reconstruction.** We train an RGB head for VGGT to reconstruct images from geometry tokens. \* indicates our implementation.

## 7. Additional Comparison Results

#### 7.1. 3D Generation

Comparison on 3D Generation with Camera Conditions. We provide the full appearance evaluation results on Co3Dv2 [43], WildRGB-D [75] and TartanAir [68] datasets in Tab. 6. Gen3R consistently surpassing existing methods across all metrics and datasets in the 1-view setting, and achieves leading performance in the 2-view setting. Additional qualitative comparisons of 3D generation are shown in Fig. 10 and Fig. 8. As observed, LVSM [19], Aether [59] and WVD [87] fail to synthesize images from novel viewpoint in 1-view setting, primarily due to poor camera con-

trollability. While Gen3C [44] can generate plausible contents, it exhibits notable shifts caused by inaccurate depth estimation. In contrast, our method produces high-fidelity results that adhere closely to the camera conditions and maintain better 3D structure, as shown in Fig. 8.

Comparison on 3D Generation without Camera Conditions. We further demonstrate our capability to generate 3D scenes from images without camera conditions. To assess this, we report the VBench Score [16, 17], focusing on I2V Subject (I2V Subj.), I2V Background (I2V BG), Aesthetic Quality (Aes.Q.), Imaging Quality (I.Q.) and Motion Smoothness (M.S.) on RealEstate10K [90] and DL3DV-10K [30] datasets. As shown in Tab. 7, our method clearly outperforms Aether [59] and WVD [87], illustrating its superior ability in generating high-quality 3D scenes.

#### 7.2. Feed-forward 3D Reconstruction

Comparison on Camera Pose Estimation. We evaluate our method on RealEstate 10K and WildRGB-D datasets for camera pose estimation, as reported in Tab. 8. Our approach achieves competitive results compared to VGGT, while notably surpassing Aether, showing the versatility and robustness of our model.

Comparison on Geometry Reconstruction. We provide additional qualitative results of feed-forward 3D reconstruction compared with VGGT [65] in Fig. 9. It can be observed that VGGT produces noticeable floaters in the reconstructed point clouds, while our method generates significantly cleaner geometry.

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 10. Qualitative Comparison of Novel View Synthesis in 1-view setting with camera conditions.

<span id="page-14-1"></span>![](_page_14_Figure_2.jpeg)

Figure 11. More Qualitative Results in 1-view setting with camera conditions.

## 7.3. Ablation Study

RGB Head for VGGT. To validate the effectiveness of our joint latents design, we train an RGB head for VGGT to enable direct RGB reconstruction from its geometry tokens V. We then compare its RGB reconstruction quality with that of Wan's RGB VAE [\[62\]](#page-10-20). The results are presented in Tab. [9.](#page-13-3) RGB VAE significantly outperforms VGGT\* , as

<span id="page-15-0"></span>![](_page_15_Figure_0.jpeg)

Figure 12. More Qualitative Results in 2-view setting with camera conditions.

<span id="page-15-1"></span>![](_page_15_Figure_2.jpeg)

Figure 13. More Qualitative Results in 1-view and 2-view settings *without* camera conditions.

VGGT is designed primarily for geometry modeling and lacks sufficient capacity for RGB feature extraction and high-fidelity appearance reconstruction. This observation also motivates our choice to decode appearance and geometry separately. By combining the strengths of both pretrained models, we achieve photorealistic video generation together with high-quality 3D structure.

# 8. More Results of Gen3R

We present additional qualitative results for both 3D generation and feed-forward 3D reconstruction in this section, including: *1) 3D Generation with Camera Conditions* (see Fig. [11](#page-14-1) and Fig. [12\)](#page-15-0); *2) Feed-Forward 3D Reconstruction* (see Fig. [14\)](#page-16-0); and *3) 3D Generation without Camera Conditions* (see Fig. [13\)](#page-15-1). We visualize the generated frames, depth maps of the sequences, and the global

<span id="page-16-0"></span>![](_page_16_Picture_0.jpeg)

Figure 14. More Qualitative Results of feed-forward reconstruction.

point clouds of the scenes.

Our method synthesizes globally consistent and photorealistic 3D scenes under diverse input conditions and effectively handles a wide range of scenarios, including indoor scenes, outdoor environments, and object-centric cases. Thanks to our design, the model exhibits strong camera controllability under conditioned settings, while also enabling free scene navigation in the absence of camera inputs. Combined with support for multiple output modalities, Gen3R provides fine-grained and coherent 3D scene generation across both constrained and unconstrained regimes.
