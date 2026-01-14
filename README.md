<div align="center">
<h1>
<span style="color:#F3E969">G</span><span style="color:#D6E67B">e</span><span style="color:#93D89A">n</span><span style="color:#69D5C9">3</span><span style="color:#55B5CA">R</span>: 3D Scene Generation Meets Feed-Forward Reconstruction
</h1>

[Jiaxin Huang](https://jaceyhuang.github.io/), [Yuanbo Yang](https://github.com/freemty), [Bangbang Yang](https://ybbbbt.com/), [Lin Ma](https://scholar.google.com/citations?user=S4HGIIUAAAAJ&hl=en),  [Yuewen Ma](https://scholar.google.com/citations?user=VG_cdLAAAAAJ), [Yiyi Liao](https://yiyiliao.github.io/)

<a href="https://xdimlab.github.io/Gen3R/"><img src="https://img.shields.io/badge/Project_Page-yellowgreen" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2601.04090"><img src="https://img.shields.io/badge/arXiv-2601.04090-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/JaceyH919/Gen3R"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

<p align="center">
  <a href="">
    <img src="./assets/teaser.png" alt="Logo" width="100%">
  </a>
</p>

<p align="left">
<strong>TL;DR</strong>: Gen3R create multi-quantity geometry with RGB from images via a unified latent space that aligns geometry and appearance.
</p>

</div>

## 🛠️ Setup
We train and test our model under the following environment: 
- Debian GNU/Linux 12 (bookworm)
- NVIDIA H20 (96G)
- CUDA 12.4
- Python 3.11
- Pytorch 2.5.1+cu124


1. Clone this repository
```bash
git clone https://github.com/JaceyHuang/Gen3R
cd Gen3R
```
2. Install packages
```bash
conda create -n gen3r python=3.11.2 -y
conda activate gen3r
pip install -r requirements.txt
```
3. (**Important**) Download pretrained Gen3R checkpoint from [HuggingFace](https://huggingface.co/JaceyH919/Gen3R) to ./checkpoints
```bash
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/JaceyH919/Gen3R ./checkpoints
```
- **Note:** At present, direct loading weights from HuggingFace via `from_pretrained("JaceyH919/Gen3R")` is not supported due to module naming errors. Please download the model checkpoint **locally** and load it using `from_pretrained("./checkpoints")`.

## 🚀 Inference
Run the python script `infer.py` as follows to test the examples
```bash
python infer.py \
    --pretrained_model_name_or_path ./checkpoints \
    --task 2view \
    --prompts examples/2-view/colosseum/prompts.txt \
    --frame_path examples/2-view/colosseum/first.png examples/2-view/colosseum/last.png \
    --cameras free \
    --output_dir ./results \
    --remove_far_points
```
Some important inference settings below:
- `--task`: `1view` for `First Frame to 3D`, `2view` for `First-last Frames to 3D`, and `allview` for `3D Reconstruction`.
- `--prompts`: the text prompt string or the path to the text prompt file.
- `--frame_path`: the path to the conditional images/video. For the `allview` task, this can be either the path to a folder containing all frames or the path to the conditional video. For the other two tasks, it should be the path to the conditional image(s).
- `--cameras`: the path to the conditional camera extrinsics and intrinsics. We also provide basic trajectories by specifying this argument as `zoom_in`, `zoom_out`, `arc_left`, `arc_right`, `translate_up` or `translate down`. In this way, we will first use [VGGT](https://github.com/facebookresearch/vggt) to estimate the initial camera intrinsics and scene scale. To disable camera conditioning, set this argument to `free`.

Note that the default resolution of our model is 560×560. If the resolution of the conditioning images or videos differs from this, we first apply resizing followed by center cropping to match the required resolution.

### More examples
<details>
<summary>Click to expand</summary>

- **First Frame to 3D**

```bash
python infer.py \
    --pretrained_model_name_or_path ./checkpoints \
    --task 1view \
    --prompts examples/1-view/prompts.txt \
    --frame_path examples/1-view/knossos.png \
    --cameras zoom_out \
    --output_dir ./results
```

- **First-last Frames to 3D**

```bash
python infer.py \
    --pretrained_model_name_or_path ./checkpoints \
    --task 2view \
    --prompts examples/2-view/bedroom/prompts.txt \
    --frame_path examples/2-view/bedroom/first.png examples/2-view/bedroom/last.png\
    --cameras examples/2-view/bedroom/cameras.json \
    --output_dir ./results
```

- **3D Reconstruction**, note that `--cameras` are ignored in this task.

```bash
python infer.py \
    --pretrained_model_name_or_path ./checkpoints \
    --task allview \
    --prompts examples/all-view/prompts.txt \
    --frame_path examples/all-view/garden.mp4 \
    --output_dir ./results
```
</details>

<!-- ## 🚗 Training
By default, we train our models using 24 H20 GPUs (96 GB VRAM each). However, the models can also be trained on other GPU configurations by appropriately adjusting the batch size. Check out the script [`train.sh`](train.sh) for details. -->

## ✅ TODO
- [x] Release inference code and checkpoints
- [ ] Release online demo
- [ ] Release training code & dataset preparation

## 🎓 Citation
Please cite our paper if you find this repository useful:

```bibtex
@misc{huang2026gen3r3dscenegeneration,
      title={Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction}, 
      author={Jiaxin Huang and Yuanbo Yang and Bangbang Yang and Lin Ma and Yuewen Ma and Yiyi Liao},
      year={2026},
      eprint={2601.04090},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.04090}, 
}
```