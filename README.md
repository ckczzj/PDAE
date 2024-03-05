# Unsupervised Representation Learning from Pre-trained Diffusion Probabilistic Models (PDAE)

This repository is official PyTorch implementation of [PDAE](https://arxiv.org/abs/2212.12990) (NeurIPS 2022).

```
@inproceedings{zhang2022unsupervised,
  title={Unsupervised Representation Learning from Pre-trained Diffusion Probabilistic Models},
  author={Zhang, Zijian and Zhao, Zhou and Lin, Zhijie},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```



## Dataset

We use the LMDB ready-to-use datasets provided by Diff-AE ([https://github.com/phizaz/diffae#lmdb-datasets](https://github.com/phizaz/diffae#lmdb-datasets)).

The directory structure should be:

```
data
├─horse
|   ├─data.mdb
|   └lock.mdb
├─ffhq
|  ├─data.mdb
|  └lock.mdb
├─celebahq
|    ├─CelebAMask-HQ-attribute-anno.txt
|    ├─data.mdb
|    └lock.mdb
├─celeba64
|    ├─data.mdb
|    └lock.mdb
├─bedroom
|    ├─data.mdb
|    └lock.mdb
```




## Download

[pre-trained-dpms](https://drive.google.com/drive/folders/1mU6zgo8WYjNmUtLXZAcsXzv8RghWN9zv?usp=share_link) (required)

[trained-models](https://drive.google.com/drive/folders/1yDeQCRQdDnrLH9HyJnHJBtOS_ZqbHSl7?usp=share_link) (optional)

You should put download in the root dicretory of this project and maintain their directory structure as shown in Google Drive.



## Install Requirements
```
pip install -r requirements.txt
```



## Training

To train DDPM, run this command:

```
bash scripts/dist_train_regular_diffusion.sh 1 0 4
```



To train PDAE, run this command:

```
bash scripts/dist_train_representation_learning.sh 1 0 4
```



To train a classifier for manipulation, run this command:

```
bash scripts/dist_train_manipulation.sh 1 0 4
```



To train a latent DPM, run this command:

```
bash scripts/dist_train_latent_diffusion.sh 1 0 4
```



You can change the config file and run path in the script file.



## Evaluation

### autoencoding

```
# modify scripts/dist_sample.sh to "${ROOT_DIR}/sampler/autoencoding_example.py"
bash scripts/dist_sample.sh 1 0 1
```

<div align=center><img src="./images/autoencoding_example_result.png" height="60"/></div>



### autoencoding evaluation

```
# modify scripts/dist_sample.sh to "${ROOT_DIR}/sampler/autoencoding_eval.py"
bash scripts/dist_sample.sh 1 0 4
```

PDAE achieves autoencoding reconstruction **SOTA** performance of **SSIM(0.994)** and **MSE(3.84e-5)** when using inferred $x_{T}$.



### denoise one step


```
# modify scripts/dist_sample.sh to "${ROOT_DIR}/sampler/denoise_one_step.py"
bash scripts/dist_sample.sh 1 0 1
```

<div align=center><img src="./images/denoise_one_step_result.png" height="120"/></div>



### posterior mean gap measure


```
# modify scripts/dist_sample.sh to "${ROOT_DIR}/sampler/gap_measure.py"
bash scripts/dist_sample.sh 1 0 4
```

<div align=center><img src="./images/gap_measure_result.png" height="300"/></div>

### interpolation


```
# modify scripts/dist_sample.sh to "${ROOT_DIR}/sampler/interpolation.py"
bash scripts/dist_sample.sh 1 0 1
```

<div align=center><img src="./images/interpolation_result.png" height="120"/></div>



### manipulation


```
# modify scripts/dist_sample.sh to "${ROOT_DIR}/sampler/manipulation.py"
bash scripts/dist_sample.sh 1 0 1
```

<div align=center><img src="./images/manipulation_result.png" height="60"/></div>



### unconditional sample

```
# modify scripts/dist_sample.sh to "${ROOT_DIR}/sampler/unconditional_sample.py"
bash scripts/dist_sample.sh 1 0 4
```

<div align=center><img src="./images/unconditional_sample_result.png" height="400"/></div>

