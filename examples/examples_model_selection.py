"""
An Introduction to the model_selection Module
=============================================

In EEG-based emotion recognition, the division of training and test sets
may differ based on research objectives. The model_selection module
addresses this need by supporting a variety of splitting strategies
considering trials, sessions, and subjects.

"""


######################################################################
# The simplest approach is ``KFold``, where the dataset is randomly
# divided into ‘k’ subsets. Each subset serves as the test set in one of
# the ‘k’ experiments, while the remaining subsets form the training set.
# The model’s performance is reported as the average result across the ‘k’
# test sets.
# 

from torcheeg.datasets import DEAPDataset
from torcheeg.model_selection import KFold

dataset = DEAPDataset(
    io_path='./tmp_out/deap',
    root_path='./tmp_in/data_preprocessed_python'
)

k_fold = KFold(split_path='./tmp_out/examples_model_selection_1/split')
for train_dataset, test_dataset in k_fold.split(dataset):
    print(len(train_dataset))
    print(len(test_dataset))


######################################################################
# Since TorchEEG automatically caches the divided indices to
# ``split_path``, there’s no need to worry about encountering different
# partitions each time the program runs. However, when you want to
# generate a new partition or use other parameters for partitioning,
# please delete the old partition at ``split_path`` or define a new
# ``split_path`` to create a new partition cache.
# 

!rm -rf ./deap


######################################################################
# To evaluate model performance under unseen trials or periods, TorchEEG
# provides ``KFoldGroupbyTrial`` and ``KFoldCrossTrial``. The former
# divides each trial into ‘k’ periods; each period serves as the test set
# in one of the ‘k’ experiments, while the remaining subsets form the
# training set. The latter takes into account inter-trial differences and
# splits each subject’s trials into ‘k’ folds, including only specific
# trials’ EEG samples in the training set.
# 

from torcheeg.model_selection import KFoldCrossTrial

k_fold = KFoldCrossTrial(split_path='./tmp_out/examples_model_selection_2/split')
for train_dataset, test_dataset in k_fold.split(dataset):
    print(len(train_dataset))
    print(len(test_dataset))

from torcheeg.model_selection import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(split_path='./tmp_out/examples_model_selection_3/split')
for train_dataset, test_dataset in k_fold.split(dataset):
    print(len(train_dataset))
    print(len(test_dataset))


######################################################################
# For the prevalent research scenario of cross-subject problems, TorchEEG
# further provides ``LeaveOneSubjectOut`` and ``KFoldCrossSubject``.
# ``KFoldCrossSubject`` divides subjects into ‘k’ folds, with the training
# set comprising only ‘k-1’ folds of subjects’ EEG samples.
# ``LeaveOneSubjectOut`` is a special case where ‘k’ equals the total
# number of subjects.
# 

from torcheeg.model_selection import LeaveOneSubjectOut

loso = LeaveOneSubjectOut(split_path='./tmp_out/examples_model_selection_3/split')
for train_dataset, test_dataset in loso.split(dataset):
    print(len(train_dataset))
    print(len(test_dataset))


######################################################################
# Additionally, we offer ``PerSubject`` versions of ``KFold``,
# ``KFoldGroupbyTrial``, and ``KFoldCrossTrial``, among others. These are
# specifically designed for research focused on individual subject model
# performance.
# 

from torcheeg.model_selection import KFoldGroupbyTrial

loso = KFoldGroupbyTrial(split_path='./tmp_out/examples_model_selection_3/split')
for train_dataset, test_dataset in loso.split(dataset):
    print(len(train_dataset))
    print(len(test_dataset))