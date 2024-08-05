# Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing (ECCV2024)

### <p align="center">[🌐Project Page](https://nju-3dv.github.io/projects/Relightable3DGaussian/) | [🖨️ArXiv](https://arxiv.org/abs/2311.16043) | [📰Paper](https://arxiv.org/pdf/2311.16043)</p>


<p align="center">
<a href="http://ygaojiany.github.io/" target="_blank">Jian Gao<sup>1*</sup></a>, <a href="https://sulvxiangxin.github.io/" target="_blank">Chun Gu<sup>2*</sup></a>, <a href="https://scholar.google.com/citations?user=VhhHLhIAAAAJ&hl=en" target="_blank">Youtian Lin<sup>1</sup></a>, <a href="http://zhuhao.cc/home/" target="_blank">Hao Zhu<sup>1</sup></a>, <a href="https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html" target="_blank">Xun Cao<sup>1</sup></a>, <a href="https://lzrobots.github.io/" target="_blank">Li Zhang<sup>2<i class="fa fa-envelope"> </i></sup></a>, <a href="https://yoyo000.github.io/" target="_blank">Yao Yao<sup>1<i class="fa fa-envelope"> </i></sup></a></h5> <br><sup>1</sup>Nanjing University <sup>2</sup>Fudan University <br> *denotes Equally contributed.
</p>

This is official implement of Relightable 3D Gaussian for the paper *Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing*.
![Alt text](media/teaser.gif)


### Installation
#### Clone this repo
```shell
git clone https://github.com/NJU-3DV/Relightable3DGaussian.git
```
#### Install dependencies
```shell
# install environment
conda env create --file environment.yml
conda activate r3dg

# install pytorch=1.12.1
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# install torch_scatter==2.1.1
pip install torch_scatter==2.1.1

# install kornia==0.6.12
pip install kornia==0.6.12

# install nvdiffrast=0.3.1
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```

#### Install the pytorch extensions
We recommend that users compile the extension with CUDA 11.8 to avoid the potential problems mentioned in [3D Guassian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```shell
# install knn-cuda
pip install ./submodules/simple-knn

# install bvh
pip install ./bvh

# install relightable 3D Gaussian
pip install ./r3dg-rasterization
```
### Data preparation
#### NeRF Synthetic Dataset
Download the NeRF synthetic dataset from [LINK](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi?usp=drive_link) provided by [NeRF](https://github.com/bmild/nerf).

#### Pre-processed DTU
For real-world DTU data, we follow the [Vis-MVSNet](https://github.com/jzhangbs/Vis-MVSNet) to get the depth maps and then filter the depth map through photometric and geometric check. We then convert the depth map to normal through [kornia](https://kornia.readthedocs.io/en/latest/geometry.depth.html). And we get perfect masks from [IDR](https://github.com/lioryariv/idr). The pre-processed DTU data can be downloaded [here](https://box.nju.edu.cn/f/d9858b670ab9480fb526/?dl=1).  

#### Pre-processed Tanks and Temples
For real-world Tanks and Temples data, we also use [Vis-MVSNet](https://github.com/jzhangbs/Vis-MVSNet) and [kornia](https://kornia.readthedocs.io/en/latest/geometry.depth.html) to get the filtered MVS depth maps and normal maps. The masks are from [NSVF](https://github.com/facebookresearch/NSVF). The pre-processed Tanks and Temples data can be downloaded [here](https://box.nju.edu.cn/f/f0b094cc22bf4934a000/?dl=1).  

#### Synthetic4Relight

Download the NeRF synthetic dataset from [LINK](https://drive.google.com/file/d/1wWWu7EaOxtVq8QNalgs6kDqsiAm7xsRh/view?usp=sharing) provided by [InvRender](https://github.com/zju3dv/InvRender).

#### Data Structure
We organize the datasets like this:
```
Relightable3DGaussian
├── datasets
    ├── nerf_synthetic
    |   ├── chair
    |   ├── ...
    ├── neilfpp
    |   ├── data_dtu
    │   │   │── DTU_scan24
    |   |   |   |── inputs
    |   |   |   |   |── depths
    |   |   |   |   |── images
    |   |   |   |   |── model
    |   |   |   |   |── normals
    |   |   |   |   |── pmasks
    |   |   |   |   |── sfm_scene.json
    │   │   │── ...
    |   ├── data_tnt
    │   │   │── Barn
    |   |   |   |── inputs
    |   |   |   |   |── depths
    |   |   |   |   |── images
    |   |   |   |   |── model
    |   |   |   |   |── normals
    |   |   |   |   |── pmasks
    |   |   |   |   |── sfm_scene.json
    │   │   │── ...
    ├── Synthetic4Relight
    |   ├── air_baloons
    |   ├── ...
    
```

#### Ground Points for composition
For multi-object composition, we manually generate a ground plane with relightable 3D Gaussian representation, which can be downloaded [here](https://box.nju.edu.cn/f/c51d9de245f04d0fb872/?dl=1). We put the *ground.ply* in the folder *./point*.

### Running
We run the code in a single NVIDIA GeForce RTX 3090 GPU (24G). To reproduce the results in the paper, please run the following code.
NeRF Synthetic dataset:
```
sh script/run_nerf.sh
```
DTU data:
```
sh script/run_dtu.sh
```
Tanks and Temples data: 
```
sh script/run_tnt.sh
```
Synthetic4Relight data: 
```
sh script/run_syn.sh
```

### Evaluating
Run the following command to evaluate Novel View Synthesis:
```
# e.g. DTU dataset
# stage 1
python eval_nvs.py --eval \
    -m output/dtu/${i}/3dgs \
    -c output/dtu/${i}/3dgs/chkpnt30000.pth

# stage 2
python eval_nvs.py --eval \
    -m output/dtu/${i}/neilf \
    -c output/dtu/${i}/neilf/chkpnt50000.pth \
    -t neilf
```
Run the following command to evaluate Relighting (for Synthetic4Relight only):
```
# e.g.
python eval_relighting_syn4.py \
    -m output/Syn4Relight/hotdog/neilf \
    -c output/Syn4Relight/hotdog/neilf/chkpnt50000.pth \
    --sample_num 384
```
### Composition and Relighting
Explicit point cloud representation facilitates composition. We recommend that users explore [cloud compare](https://www.cloudcompare.org/) to implement point cloud transformations such as scaling, translation, and rotation to composite a new scene.
For multi-object composition and relighting result demonstrated in the paper, you could run:
```
python relighting.py \
-co configs/teaser \
-e "env_map/teaser.hdr" \
--output "output/relighting/teaser_trace" \
--sample 384
```
For multi-object composition and relighting video illustrated in the project page, you could run:
```
python relighting.py \
-e env_map/composition.hdr \
-co configs/nerf_syn \
--output "output/relighting/nerf_syn" \
--sample_num 384 \
--video 

python relighting.py \
-co configs/nerf_syn_light \
-e "env_map/composition.hdr" \
--output "output/relighting/nerf_syn_light" \
--sample_num 384 \
--video 
```
For multi-scene composition video, you could run:
```
python relighting.py \
-co configs/tnt \
-e "env_map/ocean_from_horn.jpg" \
--output "output/relighting/tnt" \
--sample_num 384 \
--video
```
### GUI
We also provide a GUI for visualization. You can utilize this GUI to supervise the training process by just adding the **--gui** option to the command for training. You can also visualize the model after training by running:
```
# for 3D Gaussian
python gui.py -m output/NeRF_Syn/lego/3dgs -t render 

# for relightable 3D Gaussian
python gui.py -m output/NeRF_Syn/lego/neilf -t neilf
```
### Trying on your own data
We recommend that users reorganize their own data as neilfpp-like dataset and then optimize. Modified VisMVSNet and auxiliary scripts to prepare your own data will come soon.

### Citation
If you find our work useful in your research, please be so kind to cite:
```
@article{R3DG2023,
    author    = {Gao, Jian and Gu, Chun and Lin, Youtian and Zhu, Hao and Cao, Xun and Zhang, Li and Yao, Yao},
    title     = {Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing},
    journal   = {arXiv:2311.16043},
    year      = {2023},
}
```
