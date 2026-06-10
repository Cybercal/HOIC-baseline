# Humanoid-Object Interaction Controller Baseline

In this repo, we offer two baseline methods for simple humanoid-object interaction task. Feel free to test our designed HOI task Success Rate (SR%) in simulator for your research. If you want to apply our framework to your task setting, we also release our training code:

1. **RSS 2026:** HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model
[[project page]](https://haic-humanoid.github.io/) 
[[arXiv]](https://arxiv.org/abs/2602.11758)
[[code]](https://github.com/ldt29/HAIC)

2. **Arxiv 2026:** VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands
[[project page]](https://vaic-humanoid.github.io/) 
[[arXiv]](https://arxiv.org/abs/2606.09286)
[[code]](https://github.com/ldt29/VAIC)



## Demo

| Method | Motion Reference | Extra Sensor Dependency | Core observation |
| --- | --- | --- | --- |
| HAIC | robot-object pairs | Does not depend on extra sensors | Reference motion |
| VAIC | reference-free | Uses visual input to adapt objects | ego depth image |


<table>
<tr>
<td align="center">
  <img src="assets/demo/skateboard_sim2sim.gif" width="350"/>
  <br/>
  <b>Skateboard</b>
</td>
<td align="center">
  <img src="assets/demo/box_slopestair_sim2sim.gif" width="350"/>
  <br/>
  <b>Box via terrains</b>
</td>
</tr>
</table>


- [x] Release the HAIC policy for its sim2sim performance on HOI task.
- [ ] Release the VAIC policy for its sim2sim performance on HOI task.


## Quick Start

Use the project Python environment, then install the runtime packages:

```bash
conda create -n deploy_onnx python=3.8
conda activate deploy_onnx
pip install mujoco onnxruntime tqdm imageio imageio-ffmpeg opencv-python
```

## Usage

Run a specific task:

```bash
python scripts/deploy_haic.py --task box_plane
python scripts/deploy_haic.py --task box_slopestair
python scripts/deploy_haic.py --task box_stairslope
python scripts/deploy_haic.py --task pull_cart
python scripts/deploy_haic.py --task pull_cart_box
python scripts/deploy_haic.py --task push_cart
python scripts/deploy_haic.py --task push_wheelchair
python scripts/deploy_haic.py --task skateboard
```



## Options

```bash
python scripts/deploy_haic.py \
  --task skateboard \
  --sim_dt 0.002 \
  --control_dt 0.02 \
  --viewer_dt 0.01 \
  --num_actuators 29
```

- `--sim_dt`: MuJoCo physics timestep.
- `--control_dt`: policy/control timestep. Must be an integer multiple of `--sim_dt`.
- `--viewer_dt`: viewer/video render interval.
- `--num_actuators`: expected robot actuator count.
- `--record_video`: save a 2x-speed MP4 named `<task>_sim2sim.mp4`.
- `--video_width`: recorded video width. Default: `1280`.
- `--video_height`: recorded video height. Default: `720`.
- `--video_codec`: recorded video codec. Default: `h264`.
- `--no_real_time`: run as fast as possible.
- `--check_assets`: print task triplets and exit after validation.

Record video:

```bash
python scripts/deploy_haic.py --task skateboard --record_video
```

Fast H.264 recording helper:

```bash
python scripts/record_haic.py --task skateboard
```

Disable real-time sleeping:

```bash
python scripts/deploy_haic.py --task skateboard --no_real_time
```

Check that every configured task has its policy, motion, and XML assets:

```bash
python scripts/deploy_haic.py --check_assets
```

## File Structure

```text
Deploy_HAIC/
├── scripts/
│   ├── deploy_haic.py          ← deploy HAIC policy in MuJoCo
│   └── depoly_vaic.py          ← deploy VAIC policy in MuJoCo
├── controller/
│   ├── policy/
│   │   ├── haic/               ← HAIC ONNX policies 
│   │   └── vaic/               ← VAIC ONNX policies
│   └── motion/
│       └── mirobotA/           ← reference motion folders with meta.json and motion.npz
├── assets/
│   └── robots/
│       └── g1/
│           ├── meshes_v1/      ← G1 mesh assets used by the XML files
│           └── *.xml           ← MuJoCo task XML files
└── readme.md
```

`scripts/deploy_haic.py` loads one task triplet:

- ONNX policy from `controller/policy/haic/`
- reference motion from `controller/motion/mirobotA/`
- MuJoCo XML from `assets/robots/g1/`

The G1 XML files use mesh assets from `assets/robots/g1/meshes_v1/`.


## Citation

If you find our work useful for your research, please consider citing us:

```bibtex
@article{li2026haic,
  title = {HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model},
  author = {Li, Dongting and Chen, Xingyu and Wu, Qianyang and Chen, Bo and Wu, Sikai and Wu, Hanyu and Zhang, Guoyao and Li, Liang and Zhou, Mingliang and Xiang, Diyun and Ma, Jianzhu and Zhang, Qiang and Xu, Renjing},
  journal = {arXiv preprint arXiv:2602.11758},
  year = {2026}
}

@article{li2026vaic,
  title = {VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands},
  author = {Li, Dongting and Wu, Qianyang and Chen, Xingyu and Li, Liang and Lin, Yuhang and Wu, Sikai and Zhang, Guoyao and Zhou, Mingliang and Xiang, Diyun and Zhang, Qiang and Xu, Renjing and Ma, Jianzhu},
  journal = {arXiv preprint arXiv:2606.09286},
  year = {2026}
}

```
