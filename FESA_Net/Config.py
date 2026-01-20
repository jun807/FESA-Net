import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        

use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 32          
cosineLR = True         
n_channels = 3
n_labels = 1
epochs = 1000
img_size = 224
print_frequency = 1
save_frequency = 100
vis_frequency = 100
early_stopping_patience = 100

pretrained = True
use_mobile_blocks = True

#dataset_name = 'ISIC2017'
#dataset_name = 'CVC-ClinicDB'
#dataset_name = 'ISIC2018'
#dataset_name = 'Covid'
dataset_name = 'ISIC2018'
#dataset_name = 'Kvasir-SEG'
#dataset_name = 'BUSI'


#task_name = 'GlaS_exp3'
task_name = 'ISIC18_exp1'
#task_name = 'Clinic_exp3'
#task_name = 'ISIC17_exp2'
#task_name = 'Kvasir_exp1'
#task_name = 'Covid_exp1'
#@task_name = 'ISIC18_exp1'
#task_name = 'BUSI_exp2'
#task_name = 'ISIC18_exp2'


learning_rate = 1e-3
batch_size = 16

model_name = 'FESA_Net'
#model_name = 'SwinPA'
#model_name = 'SwinUnet'
#model_name = 'SMESwinUnet'
#model_name = 'UCTransNet'
#model_name = 'UNet_base'
#model_name = 'MultiResUnet1_32_1.67'

test_session = "session"         #



train_dataset = '/mnt/d/论文实验/Datasets/'+ dataset_name+ '/Train_Folder/'
val_dataset = '/mnt/d/论文实验/Datasets/'+ dataset_name+ '/Val_Folder/'
test_dataset = '/mnt/d/论文实验/Datasets/'+ dataset_name+ '/Test_Folder/'

session_name       = 'session'  #time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'



##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config


