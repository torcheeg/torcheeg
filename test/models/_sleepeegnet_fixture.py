"""External native oracle; no restricted author functions are redistributed."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.layers.core import Dense
    from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

    source = Path(args.source)
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == '496bdcb027b3dbc744f249d53f82388a833617204c881223074fd16e8b07f6bb'
    tree = ast.parse(data, filename=str(source))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ('flatten', 'build_firstPart_model', 'build_network', 'build_whole_model')
    ]
    assert len(tree.body) == 4
    scope = dict(tf=tf, np=np, Dense=Dense, beam_search_decoder=beam_search_decoder)
    exec(compile(tree, str(source), 'exec'), scope)
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for bidirectional in [False, True]:
        with tf.Graph().as_default():
            tf.set_random_seed(1701)
            hparams = tf.contrib.training.HParams(
                batch_size=2,
                num_units=128,
                embed_size=10,
                input_depth=3000,
                n_channels=100,
                bidirectional=bidirectional,
                use_attention=True,
                lstm_layers=2,
                attention_size=64,
                beam_width=4,
                use_beamsearch_decode=False,
                max_time_step=10,
                output_max_length=12,
                akara2017=True,
            )
            inputs = tf.placeholder(tf.float32, [2, 10, 3000])
            dec_inputs = tf.placeholder(tf.int32, [2, 11])
            targets = tf.placeholder(tf.int32, [2, 11])
            logits, predictions, loss, _optimizer, state = scope['build_whole_model'](
                hparams,
                {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, '<SOD>': 5, '<EOD>': 6},
                inputs,
                targets,
                dec_inputs,
                1.0,
            )
            variables = tf.trainable_variables()
            gradients = [
                tf.convert_to_tensor(g)
                for g in tf.gradients(loss, [inputs] + variables)
            ]
            update = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
            alignment = state.alignment_history.stack()
            rng = np.random.RandomState(102)
            x = (rng.normal(size=(2, 10, 3000)) * 0.2).astype(np.float32)
            y = rng.randint(0, 5, size=(2, 10)).astype(np.int32)
            decoder = np.concatenate((np.full((2, 1), 5, np.int32), y), 1)
            target = np.concatenate((y, np.full((2, 1), 6, np.int32)), 1)
            feed = {inputs: x, dec_inputs: decoder, targets: target}
            with tf.Session(
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=2, inter_op_parallelism_threads=2
                )
            ) as session:
                session.run(tf.global_variables_initializer())
                weights = session.run(variables)
                scores, tokens, objective, attention, grads = session.run(
                    [logits, predictions, loss, alignment, gradients], feed
                )
                session.run(update, feed)
                after = session.run(variables)
                scores_after, tokens_after = session.run([logits, predictions], feed)
            metadata = dict(
                revision='288ee50700ef34780c6e056a884cf22308aa6de2',
                source_sha256=digest,
                neural_functions_modified=False,
                source_functions=[
                    'flatten',
                    'build_firstPart_model',
                    'build_network',
                    'build_whole_model',
                ],
                tensorflow=tf.__version__,
                bidirectional=bidirectional,
                parameter_count=sum(v.size for v in weights),
                variable_names=[v.name[:-2] for v in variables],
                dropout=0.0,
                learning_rate=0.01,
            )
            payload = dict(
                metadata=np.array(json.dumps(metadata)),
                input=x,
                labels=y.astype(np.int64),
                decoder_inputs=decoder.astype(np.int64),
                targets=target.astype(np.int64),
                logits=scores,
                tokens=tokens,
                loss=objective,
                attention=attention.transpose(1, 0, 2),
                input_gradient=grads[0],
                logits_after=scores_after,
                tokens_after=tokens_after,
            )
            for i, (v, g, a) in enumerate(zip(weights, grads[1:], after)):
                payload['state_' + str(i)] = v
                payload['gradient_' + str(i)] = g
                payload['delta_' + str(i)] = a - v
            filename = dest / (
                'bidirectional.npz' if bidirectional else 'unidirectional.npz'
            )
            np.savez_compressed(filename, **payload)
            print(
                'FIXTURE_OK',
                bidirectional,
                metadata['parameter_count'],
                len(variables),
                scores.shape,
                tokens.shape,
                float(objective),
                flush=True,
            )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


if __name__ == '__main__':
    main()
