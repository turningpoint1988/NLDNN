"""Params for NLDNN-AT."""
import torch
import os.path as osp

# params for dataset and data loader
batch_size = 500
seq_size = 600

# params for source dataset
src_dataset = "Human"

# params for target dataset
tgt_dataset = "Mouse"

# load the number of negative data
root = osp.dirname(__file__)
neg_count = {}
with open(osp.join(root, 'count.txt'), 'r') as f:
    for line in f:
        line_split = line.strip().split()
        key = line_split[0]
        neg_count[key] = {
            'human.tr.neg': int(line_split[1]), 'human.te.neg': int(line_split[2]), 'human.va.neg': int(line_split[3]),
            'mouse.tr.neg': int(line_split[4]), 'mouse.te.neg': int(line_split[5]), 'mouse.va.neg': int(line_split[6])
        }

# params for training network
gpu = '0'
num_epochs_pre = 60
eval_step_pre = 30
num_epochs = 3
eval_step = 1
manual_seed = 666
if torch.cuda.is_available():
    torch.cuda.manual_seed(manual_seed)
    device = torch.device("cuda:" + gpu)
else:
    device = torch.device("cpu")
    torch.manual_seed(manual_seed)

trial = 3
d_train_step = 1400
ratio = [0, 0.001, 0.1, 0.5, 1]
clip_value = [0.02, 0.05]
weight = 1
alpha = 2
# params for optimizing models
s_learning_rate = 1e-3
d_learning_rate = 1e-4
t_learning_rate = 1e-5
beta1 = 0.5
beta2 = 0.9
weight_decay = 0.00001
