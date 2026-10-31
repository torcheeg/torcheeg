SleepEEGNet
===========

:class:`torcheeg.models.SleepEEGNet` implements the complete single-channel
`published CNN/sequence-to-sequence architecture <https://arxiv.org/abs/1903.02108>`_
in PyTorch. Both temporal CNN branches, stacked encoder, learned token
embedding, stacked decoder, input-fed Luong attention and output projection
are included. It is not a CNN-only classifier or a recurrent encoder with a
replacement linear head.

The separately obtained
`author implementation <https://github.com/MousaviSajad/SleepEEGNet/tree/288ee50700ef34780c6e056a884cf22308aa6de2>`_
has academic/non-commercial source terms. No restricted author source or
pretrained weights are redistributed here. The PyTorch implementation is
newly written; native numerical verification executes complete original
functions from an explicit external source file in their TensorFlow 1 runtime.
This does not establish paper/Arena protocol or published-score reproduction.

Representation and full architecture
------------------------------------

Input is floating ``[batch, epochs, 1, chunk_size]``. Defaults represent ten
30-second, single-channel epochs at 100 Hz, but the module itself does not
resample, filter or infer units. Keep each context within a single physical
recording, with no unscored gaps. There is no input padding mask.

The complete CNN uses small/large initial kernels 50/400, strides 6/50,
64/128 channels, inner kernels 8/6, both pooling stacks and all three dropouts.
Time precedes channels in each flattened branch. With 3,000 samples, there
are 3,072 CNN features per epoch. Recurrent cells retain a single i/j/f/o
kernel, one bias and the explicit unit forget bias. No extra recurrent bias
or bidirectional-layer cross-connection is introduced.

The default ``bidirectional=False`` follows the released code and has
2,642,438 parameters / 28 learned state tensors. The paper describes a
bidirectional encoder; choose ``bidirectional=True`` explicitly for the full
5,107,206-parameter / 32-state variant. This uses two independently stacked
encoder directions and concatenates their outputs and every layer's final
cell/hidden states. The decoder width doubles. A standard multi-layer
bidirectional LSTM mixes directions at higher layers and is not equivalent.

Training tokens and objective
-----------------------------

Sleep classes are zero through ``num_classes-1``. SOD is ``num_classes``;
EOD is ``num_classes+1``. The default vocabulary has seven tokens, including
five stages. SOD/EOD are not sleep stages.

.. code-block:: python

   import torch
   from torcheeg.models import SleepEEGNet

   model = SleepEEGNet(bidirectional=True)
   eeg = torch.randn(2, 10, 1, 3000)
   stages = torch.randint(0, 5, (2, 10))
   decoder_inputs, targets = model.prepare_targets(stages)
   logits = model(eeg, decoder_inputs)  # [2, 11, 7]
   loss = model.compute_loss(logits, targets)
   loss.backward()

``prepare_targets`` prepends SOD to decoder inputs and appends EOD to targets.
``forward`` teacher-forces every supplied token. It optionally returns each
step's normalized encoder attention via ``return_attention=True``.

``compute_loss`` preserves the released class-balancing equation: within each
sequence, average CE separately for each present token class, sum those class
means, then average sequences. EOD participates and absent classes contribute
zero. Add ``l2_weight * 0.5 * sum(parameter ** 2)`` over non-bias variables;
the default coefficient is ``0.001``. This differs from ordinary token CE,
class weights estimated over the whole dataset, or regularization of biases.
Do not double-count the penalty with optimizer weight decay.

Autoregressive inference
------------------------

.. code-block:: python

   model.eval()
   generated = model.generate(eeg)  # No stage labels supplied.
   tokens = generated['tokens']
   lengths = generated['lengths']

``generate`` begins at SOD and feeds its own previous prediction. It returns
tokens, logits, alignments and first-EOD-inclusive lengths. By default the
maximum is ``input_epochs+2``. All rows keep advancing until every row has
ended or the maximum is reached, matching the reference's greedy decoder.
Ignore outputs after each row's length. Early EOD, SOD or any missing stage
must not be silently scored as a valid sleep class. The example counts them
as incorrect and reports valid-stage coverage. Beam search is not exposed;
the original optional beam branch has a different output interface and is
not claimed verified. ``generate`` disables gradients; call ``eval()`` to
also disable CNN dropout.

Checkpoints and complete native validation
------------------------------------------

Use ordinary strict PyTorch ``state_dict`` checkpoints and preserve model
configuration. ``native_state_dict`` and ``load_native_state_dict`` convert
all exact TensorFlow names/layouts, including convolution axes and recurrent
gate kernels. Strict import rejects missing/additional/duplicate/wrong-shaped
variables. Native optimizer slots are separate. No guessed key, partial-layer
or incompatible-direction conversion is performed.

The test helper ``test/models/_sleepeegnet_fixture.py`` requires an explicitly
supplied author source file with verified SHA256. It executes the unchanged
complete ``flatten``, ``build_firstPart_model``, ``build_network`` and
``build_whole_model`` functions, including both CNNs, encoder, attention,
teacher decoder, greedy decoder and original training loss. Data loading,
plotting and the training driver are not executed. No neural component is
stubbed or replaced. Run it separately under Python 3.7 / TensorFlow 1.15.5::

   python test/models/_sleepeegnet_fixture.py \
       --source /path/to/seq2seq_sleep_sleep-EDF.py --output-dir /tmp/sleepeegnet

Then validate in the current TorchEEG environment::

   TORCHEEG_SLEEPEEGNET_FIXTURES=/tmp/sleepeegnet \
       python -m unittest test.models.test_sleepeegnet \
       test.models.test_sleepeegnet_native test.examples.test_sleepedf_sleepeegnet

Both full default and bidirectional variants are checked against every native
variable, teacher logits, attention, original class-balanced loss, greedy
tokens, input/all parameter gradients and complete SGD updates. Dropout is
disabled only for cross-framework arithmetic because the random streams
differ. Independent tests cover dropout/optimizer/RNG restoration, recurrent
equations, finite differences, physical isolation and token boundaries.
TensorFlow is never required to use the production PyTorch model.

Real-data workflow
------------------

See :doc:`../auto_examples/examples_sleepedf_sleepeegnet`. The independent
Sleep-EDF example uses explicit volts-to-microvolts conversion, whole-record
0.3--40 Hz filtering and 100 Hz / 30-second epochs. Sample contexts across
recordings by time, keep all held-out-subject nights separate and fit channel
statistics only from training subjects. It trains the complete bidirectional
network with fixed-epoch SGD, then runs label-free greedy test decoding.
Save full model/optimizer/RNG, statistics, hashes, physical IDs, stage counts
and predictions atomically without overwrite. This short offline teaching
run differs from the paper's full data preparation, optimizer and evaluation.
