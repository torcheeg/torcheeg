torcheeg.trainers
===============================

.. contents:: Extensive trainers used to implement different training strategies, such as vanilla classification, domain adaption, etc.
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torcheeg.trainers

Basic Classification
----------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: trainertemplate.rst

   ClassifierTrainer

Cross-domain Classification
----------------------------------

The individual differences and nonstationary of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects, since the training set and test set come from different data distributions. Domain adaptation is used to address the problem of distribution drift between training and test sets and thus achieves good performance in subject-independent (cross-subject) scenarios. 

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: trainertemplate.rst

   CORALTrainer
   DDCTrainer
   DANTrainer
   JANTrainer
   ADATrainer
   DANNTrainer
   CenterLossTrainer

Imbalance Learning for Classification
----------------------------------

EEG emotion datasets have the problem of sample class imbalance, and imbalance learning can be used to solve the class imbalance problem in emotion recognition tasks.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: trainertemplate.rst

   LALossTrainer
   LDAMLossTrainer
   EQLossTrainer
   FocalLossTrainer
   WCELossTrainer

EEG Generation
----------------------------------------

Data scarcity and data imbalance are one of the important challenges in the analysis of EEG signals. TorchEEG provides different types of generative model trainers to help train generative models to augment EEG datasets. The trainer starting with "C" represents the trainer of the category conditioned generative models, allowing the user to control the category of the generated EEG signal. The others generate samples close to real EEG signals randomly without controlling the category.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: trainertemplate.rst

   BetaVAETrainer
   CBetaVAETrainer
   WGANGPTrainer
   CWGANGPTrainer

Self-supervised Algorithm for Pre-training
----------------------------------------
As the cost of data collection decreases, the difficulty of obtaining unlabeled data is greatly reduced. How to use unlabeled data to train the model so that the model can learn task-independent general knowledge on large-scale datasets has attracted extensive attention. In natural language processing and computer vision, self-supervised models have made continuous progress by building pretext tasks to learn good language or visual representations. Today, self-supervised learning algorithms are also being tested for EEG analysis to train larger models.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: trainertemplate.rst

   SimCLRTrainer
   BYOLTrainer