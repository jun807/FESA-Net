# FESA-Net
Medical image segmentation is pivotal for clinical diagnosis and treatment plan
ning. Although U-Net and its variants have achieved remarkable success, their
performance still can not meet the needs in practical applications. This critical
issue primarily stems from two inherent drawbacks: the ineffective aggregation
of multi-scale features and the inadequate representation of fine-grained feature.
To address these two issues, we propose the Frequency-Enhanced Scale-Aware
Fusion Network (FESA-Net). Specifically, to achieve adaptive multi-scale feature
fusion, we design Cross-Layer Attention-Guided Fusion Module (CLFM) CLFM
utilizes coordinate spatial and channel attention derived from adjacent encoder
layers as guidance to aggregate multi-level features, thereby capturing positional
saliency and inter-layer correlations. Following that, we introduce the Multi-scale
Spectral Refinement Module (MSRM) to refine the fine-grained feature represen
tation. MSRMintegrates multi-scale convolutions to capture spatial details, while
employing patch-wise attention within channel groups and performing patch-wise
spectral modulation to adaptively modulate the frequency components effective
for target feature details. Extensive experiments demonstrate that FESA-Net
achieves an optimal balance between accuracy and efficiency, outperforming
mainstream methods on multiple public datasets with minimal computational
overhead (3.09 M parameters, 1.05 GFLOPs).

## Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:

* ISIC-2018 (dermoscopy, with 2,594 images)
* Kvasir-SEG (endoscopy, with 1,000 images)
* BUSI (breast ultrasound, with 437 benign and 210 malignant images)
* CVC-ClinicDB (colonoscopy, with 612 images)
* GlaS (gland, with 165 images)

In GlaS dataset, we split the dataset into a training set of 85 images and a test set of 80 images.

In ISIC 2018 dataset, we adopt the official split configuration, consisting of a training set with 2,594 images, a validation set with 100 images, and a test set with 1,000 images.

For each dataset, the images are randomly split into training, validation, and test sets with a ratio of 6:2:2.

The dataset path may look like:
```
/The Dataset Path/
├── ISIC-2018/
    ├── Train_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Val_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Test_Folder/
        ├── img
        ├── labelco
```
## Usage  
### Installation  
```
https://github.com/jun807/FESA-Net.git
conda create -n fesa_net python=3.8
conda activate fesa_net
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
### Training 
```
python train_model.py
```
To run on different setting or different datasets, please modify:
batch_size, model_name, and task_name in Config.py.

### Evaluation 
```
python test_model.py
```
