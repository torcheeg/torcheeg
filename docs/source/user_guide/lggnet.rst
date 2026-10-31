LGGNet with an independent DEAP workflow
========================================

:class:`torcheeg.models.LGGNet` accepts floating EEG shaped
``[batch, 1, electrodes, time]`` and returns class logits. Keep DEAP's native
32 EEG row order when using ``DEAP_GENERAL_REGION_LIST``: these are channel
indices, not instructions to concatenate rows into region order.

The built-in implementation retains the complete published TorchEEG v1.1.3
forward graph and state keys: multiscale temporal branches, CBAM, local filters,
regional aggregation, learned global graph and classifier. Querying
``feature_dim`` after initialization uses the learned filter width and leaves
BatchNorm, random streams, device and dtype unchanged. The legacy constructor
path is retained for checkpoint and initialization compatibility. Some legacy
convolution/BatchNorm parameters are used only during construction and receive
no task gradients; they remain in the state dictionary. This is not a claim
of numerical equivalence to the original paper's author implementation.

Independent training example
-----------------------------

``examples/examples_deap_lggnet.py`` requires authorized, trusted DEAP Python
pickles and has no NNI or model-selector dependency. Provider EEG is microvolts
at 128 Hz, already filtered at 4--45 Hz. One first-six-second stimulus window
per trial begins after the three-second baseline. No additional filter or
baseline subtraction is applied. The last sorted subject is held out. Explicitly sort each partition by subject
and trial before training; export the physical trial IDs. Fit
per-channel mean and population standard deviation over training subjects and
time only. Valence below five is class zero; at least five is class one.

Use fixed training epochs; the held-out labels are used only for evaluation,
not checkpoint selection. The default full network trains with Adam and cross
entropy. The atomic export refuses to overwrite an existing file and contains
complete model/optimizer/random states, training statistics, channel/region
indices, physical units and offset, source hashes, target/logit pairs and names
of parameters without task gradients. Strict reload must reproduce predictions.
CLI resume is not implemented. This short example is a teaching protocol,
not paper/Arena benchmark reproduction, and does not establish a leaderboard
score or pretrained foundation-model capability.

.. code-block:: bash

   python examples/examples_deap_lggnet.py \
       --root-path ./data_preprocessed_python \
       --epochs 1 --output-path ./deap-lggnet.pt
