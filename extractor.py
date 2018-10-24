import numpy as np
import scipy.io as sio
import os
import regressor_opt as regs

# the contents of the following function should be customized to fit the convention
# of the dataset you use
def handle_subj_ops(subj):
    subj_file = os.path.join(src_fold, 'subj' + format(subj, '03') + '.mat')
    data = sio.loadmat(subj_file)['data']
    # eliminate unwanted regions
    data = data[:, mask]

    subj_dest_fold = os.path.join(dst_fold, 'subj' + format(subj, '03'))
    os.mkdir(subj_dest_fold)
    return data

def extract_subj_mesh(subj, data, alpha, lambda_, epochs):

    weights = np.zeros((np.sum(task_win_counts), 8100))
    mesh_labels = np.zeros((np.sum(task_win_counts), ))

    ind = 0
    # where to save the created brain graphs and their corresponding labels
    subj_dest_fold = os.path.join(dst_fold, 'subj' + format(subj, '03'))
    for task in range(7):
        task_data = data[labels[:,0] == task +1, :]
        # iterate over non-overlapping windows
        for win in range(task_win_counts[task]):
            # get the data within the interval
            win_data = task_data[win * W : (win + 1) *W, : ]
            # get directed brain graph for the window
            weights[ind, :] = regs.minimize_dir(win_data, epochs, alpha, lambda_, debug_choice=1)
            # the label for the brain graph is the current task, +1 for 1-indexed notation
            mesh_labels[ind, ] = task+1
            ind += 1
    # save the brain networks of the subject
    f = os.path.join(subj_dest_fold, 'alp_'+ str(alpha) + 'lambda_'+str(lambda_) + '.mat' )
    # needed data for classification, labels and features(brain grahps' weights in our case)
    d = {}
    d['weights'] = weights
    d['labels'] = mesh_labels
    sio.savemat(f, d)

# source of HCP dataset or any fMRI dataset with similar data structure
src_fold  = '/home/baris/Documents/fmri_data/HCP/data'
label_file = '/home/baris/Documents/fmri_data/HCP/labels.mat'
dst_fold = '../intermediate/'

labels = sio.loadmat(label_file)['labels']

# We don't use all 116 anatomic regions defined by the AAL brain template
# to exclude Cerebellum and Vermis the following mask is defined
# subregions in these two regions mapped to False and the rest is to True
mask = np.zeros(116, dtype=bool)
mask[0:8] = True
mask[26:108] = True

# window length
W = 40
task_win_counts = np.zeros(7, dtype=int)

# how many windows are there in each task (there are 7 tasks in HCP dataset with
# varying durations
for task in range(7):
    task_win_counts[task] = np.sum(labels == task+1)/W

# learning rate
alpha = 0.00005
# set of regularization hyperparameters to test
lambdas_ = [0, 32., 64., 128., 256., 512.]
epoch_counts = [ 20000, 10000, 3000, 2000, 2000, 1000 ]
# 808 subject
subjects = range(1,809)

for subj in subjects:
    # read and create necessary folders
    subj_data = handle_subj_ops(subj)
    for lambda_ind in range(len(lambdas_)):
        print subj, lambdas_[lambda_ind]
        extract_subj_mesh(subj, subj_data, alpha, lambdas_[lambda_ind], epoch_counts[lambda_ind])
