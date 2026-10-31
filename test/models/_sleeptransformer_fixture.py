"""Run the complete unchanged pinned TensorFlow 1 SleepTransformer."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import numpy as np
    import tensorflow as tf

    root = Path(args.reference)
    folder = root / 'shhs/scratch_training/sleeptransformer'
    hashes = {
        'sleeptransformer.py': '12c34940b8c3866797debcd82c0e781ec38aeabe73cd07945dae7e387b8018d4',
        'config.py': '284bca689d5dbcc0e307ad76bdc0548b41a969230dd5d5e0b176b72fc81fe079',
        'modules.py': '7de287ee19096e244f7b1c1da7e8d4936f150c835e6f2a200b938b7ad1246289',
        'nn_basic_layers.py': 'cbb75eafde3eb8a696311fb46194577a8a96c6c134aa149e05ed373882ffb61d',
        'transformer_encoder.py': 'a580c5999a8b0e51dc89b6b1543c7c431ca54f698cbae9baf845f01355408932',
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    sys.path.insert(0, str(folder))
    from config import Config
    from sleeptransformer import SleepTransformer

    config = Config()
    config.frm_fc_dropout = config.frm_attention_dropout = config.seq_fc_dropout = (
        config.seq_attention_dropout
    ) = config.fc_dropout = 0.0
    tf.set_random_seed(1701)
    model = SleepTransformer(config)
    variables = tf.trainable_variables()
    gradients = tf.gradients(model.loss, [model.input_x] + variables)
    update = tf.train.GradientDescentOptimizer(0.01).minimize(model.loss)
    rng = np.random.RandomState(102)
    x = (rng.normal(size=(2, 20, 29, 128, 1)) * 0.2).astype(np.float32)
    y = rng.randint(0, 5, size=(2, 20)).astype(np.int64)
    feed = {
        model.input_x: x,
        model.input_y: np.eye(5, dtype=np.float32)[y],
        model.istraining: False,
    }
    with tf.Session(
        config=tf.ConfigProto(
            intra_op_parallelism_threads=2, inter_op_parallelism_threads=2
        )
    ) as session:
        session.run(tf.global_variables_initializer())
        state = session.run(variables)
        scores, loss, ce, context, grads = session.run(
            [
                model.scores,
                model.loss,
                model.output_loss,
                model.attention_weight,
                gradients,
            ],
            feed,
        )
        session.run(update, feed)
        after = session.run(variables)
        scores_after = session.run(model.scores, feed)
    metadata = dict(
        revision='26167f26ddd8bb59ce787e56655c6e99ba446d2d',
        tensorflow=tf.__version__,
        hashes=hashes,
        dropout=0.0,
        learning_rate=0.01,
        parameter_count=sum(v.size for v in state),
        variable_names=[v.name[:-2] for v in variables],
        neural_source_modified=False,
    )
    payload = dict(
        metadata=np.array(json.dumps(metadata)),
        input=x,
        targets=y,
        logits=scores,
        loss=loss,
        ce=ce,
        frame_context=context,
        input_gradient=grads[0],
        logits_after=scores_after,
    )
    for i, (v, g, a) in enumerate(zip(state, grads[1:], after)):
        payload['state_' + str(i)] = v
        payload['gradient_' + str(i)] = g
        payload['after_' + str(i)] = a
    np.savez_compressed(args.output, **payload)
    for name, digest in hashes.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    print(json.dumps(metadata, sort_keys=True))
    print('NATIVE_FIXTURE_OK', scores.shape, float(loss), float(ce))


if __name__ == '__main__':
    main()
