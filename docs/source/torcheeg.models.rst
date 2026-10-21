torcheeg.models

.. contents:: Extensive baseline method reproduction.
   :depth: 2
   :local:
   :backlinks: top
   
.. currentmodule:: torcheeg.models

Convolutional Neural Networks
----------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   EEGNet
   ShallowConvNet
   DeepConvNet
   FBCCNN
   FBCNet
   FBMSNet
   MTCNN
   STNet
   TSCeption
   CCNN
   SSTEmotionNet
   TinySleepNet
   DeepSleepNet
   USleep
   TSLANet
   CSPNet

Recurrent Neural Networks
----------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   GRU
   LSTM

Graph Neural Networks
----------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   DGCNN
   LGGNet
   RGNN
   GIN

Transformer
----------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   SimpleViT
   ArjunViT
   VanillaTransformer
   ViT
   ATCNet
   Conformer
   LaBraM
   TCNet
   MHANet
   DARNet
   CTNet
   EEGPT
   EEGPTClassifier
   EEGPTRegression
   EEGPTEmbedding
   EEGPTReconstructor
   EEGPTPredictor
   EEGPTFinetuneClassifier
   EEGPTResample
   EEGPTInputAdapter
   EEGPTAdaptedModel
   LUNA
   LUNAClassifier
   REVE
   REVEClassifier
   CodeBrain
   CodeBrainClassifier
   CodeBrainTokenizer
   EEGMamba
   EEGMambaClassifier
   TFMTokenizer
   TFMEncoder
   TFMClassifier
   LEAD
   LEADClassifier
   NeuroRVQ
   NeuroRVQClassifier
   SleepFM
   SleepFMClassifier
   SleepFMStaging
   SleepFMDiagnosis
   SleepFMSurvivalObjective
   BENDR
   BENDREncoder
   BENDRContextualizer
   BENDRClassifier
   BIOT
   BIOTClassifier
   CBraMod
   CBraModClassifier
   CBraModRegression
   CBraModEmbedding


EEGPT parameter-efficient tuning
--------------------------------

.. autofunction:: add_eegpt_adapters

.. autofunction:: add_eegpt_prefix

.. autofunction:: add_eegpt_lora

Signal processing layers
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   FourierResample

Generative Adversarial Network
----------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   BGenerator
   BDiscriminator
   BCGenerator
   BCDiscriminator
   EEGfuseNet
   EFDiscriminator   

Variational Auto Encoder
----------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   BEncoder
   BDecoder
   BCEncoder
   BCDecoder

Normalization Flow
----------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   BGlow
   BCGlow

Diffusion Models
----------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: modeltemplate.rst

   BUNet
   BCUNet
