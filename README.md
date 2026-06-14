# RSR-NF

Implementation of *RSR-NF: Neural Field Regularization by Static Restoration Priors for Dynamic Imaging* ([IEEE MLSP 2025](https://ieeexplore.ieee.org/document/10535218), [arXiv]([https://arxiv.org/abs/2304.03483](https://arxiv.org/pdf/2503.10015)), [Video](https://youtu.be/_Dq---J83Q4?si=rvBiybS3WIgsIGpr))

*Berk Iskender, Sushan Nakarmi, Nitin Daphalapurkar, Marc L. Klasky, Yoram Bresler*

Dynamic imaging involves the reconstruction of a spatio-temporal object at all times using its undersampled measurements. In particular, in dynamic computed tomography (dCT), only a single projection at one view angle is available at a time, making the inverse problem very challenging. Moreover, ground-truth dynamic data is usually either unavailable or too scarce to be used for supervised learning techniques. To tackle this problem, we propose RSR-NF, which uses a neural field (NF) to represent the dynamic object and, using the Regularization-by-Denoising (RED) framework, incorporates an additional static deep spatial prior into a variational formulation via a learned restoration operator. We use an ADMM-based algorithm with variable splitting to efficiently optimize the variational objective. We compare RSR-NF to three alternatives: NF with only temporal regularization; a recent method combining a partially-separable low-rank representation with RED using a denoiser pretrained on static data; and a deep-image prior-based model. The first comparison demonstrates the reconstruction improvements achieved by combining the NF representation with static restoration priors, whereas the other two demonstrate the improvement over state-of-the art techniques for dCT. 

<img width="3090" height="1620" alt="image" src="https://github.com/user-attachments/assets/f52ed828-aa08-4ede-93b0-42272a5b3c10" />

## Parameter configurations
RSR-NF training parameters are stored and can be modified at ```configs/red_nf_train_cfg.yaml```.

The default configuration is for the dynamic walnut object with total number of views P=256. Configurations for other settings are reported in the supplementary material of the manuscript.

## RSR restoration operator
Pre-trained DnCNN restoration operators for the dynamic walnut object is provided in ```data/restoration_operator```. 
The denoiser code can be found in ```red_psm_models.py```. 
If required, denoisers pre-trained on different objects/distributions can also be incorporated using the denoiser definition in ```models.py???```.

## Forward Model
To download tomographic forward models (measurement operators) for different total number of measurements (32, 64, 128, and 256), run
```shell
cd forward_model
bash forward_model_download_P32.sh
bash forward_model_download_P64.sh
bash forward_model_download_P128.sh
bash forward_model_download_P256.sh
```
Different forward models for different imaging modalities with dimensions ```[measurement size x image size x total number of measurements]``` can also be used with RSR-NF.

## Citation
If you find RSR-NF useful for your research, please cite:

*B. Iskender, Y. Bresler, S. Nakarmi, N. Daphalapurkar and M. L. Klasky, "RSR-NF: Neural Field Regularization by Static Restoration Priors for Computed Dynamic Imaging," 2025 IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP), Istanbul, Turkiye, 2025, pp. 1-6, doi: 10.1109/MLSP62443.2025.11204251.*

```
@inproceedings{iskender2025rsr,
  title={RSR-NF: Neural Field Regularization by Static Restoration Priors for Computed Dynamic Imaging},
  author={Iskender, Berk and Bresler, Yoram and Nakarmi, Sushan and Daphalapurkar, Nitin and Klasky, Marc L},
  booktitle={2025 IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```

## Contact
In case of any questions, feel free to contact via email: Berk Iskender, berk.iskender9@gmail.com.
