import pickle
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from constants import ROOT_DIR_PATH, TMP_ARTIFACTS_PATH
from tasks.operators.mta.cli_utils import run_console_tool
from tasks.operators.mta.task import MTATask
from utils import logger

sys.path.insert(0, f'{ROOT_DIR_PATH}/tasks/operators/tpr_toolkit')

import tasks.operators.tpr_toolkit as tpr

from tasks.common.error_estimator import BinaryVectorErrorEstimator


def new_empty_placeholder(numbers_quantity, batch_size, bits_per_number, bits_per_vector):
    return np.zeros(
        (
            batch_size,
            numbers_quantity * (bits_per_number + 1),  # we sum N numbers, each followed by special marker
            bits_per_vector
        ), dtype=np.float32)


def single_tpr_len(two_tuple_largest_scale_size) -> int:
    first_tuple = tpr.model_2_tuple.Model2Tuple(term_index=0, alpha=0,
                                                linguistic_scale_size=two_tuple_largest_scale_size)
    encoded_tuple, encoder = tpr.model_2_tuple.encode_model_2_tuple(first_tuple)
    flattened_encoded_tuple = tpr.flattenize_per_tensor_representation(encoded_tuple)
    return len(flattened_encoded_tuple)


def log_generated_mta_samples(samples: List[np.ndarray], results, bits_per_number, numbers_quantity,
                              linguistic_scale_size, is_verbose=True, full_check=False, encoder=None, decoder=None,
                              mta_encoding=None):
    assert mta_encoding, 'MTA Encoding type should be known for logging it'
    if full_check:
        to_check = samples
    else:
        to_check = samples[:1]
    for sample_index, sample in enumerate(to_check):
        tprs = []
        for tpr_index in range(numbers_quantity):
            number_starts_at = tpr_index * (bits_per_number + 1)
            number_finishes_at = number_starts_at + bits_per_number
            tprs.append(sample[number_starts_at:number_finishes_at][:, 0])
        res_tpr = results[sample_index][:, 0]

        model_2_tuple_has_weights = mta_encoding == MTATask.MTAEncodingType.full
        model_tuples = [
            tpr.model_2_tuple.decode_model_2_tuple_tpr(i,
                                                       decoder=decoder,
                                                       model_2_tuple_has_weights=model_2_tuple_has_weights)[0]
            for i in tprs]
        res_tuple = tpr.model_2_tuple.decode_model_2_tuple_tpr(res_tpr, decoder=decoder,
                                                               model_2_tuple_has_weights=model_2_tuple_has_weights)[0]

        if is_verbose:
            input_model_tuple_str = ', '.join(map(str, model_tuples))
            logger.info(f'B: {sample_index}. Generated sample {{ {input_model_tuple_str} }} -> {{ {res_tuple} }}')
        assert tpr.model_2_tuple.aggregate_model_tuples(model_tuples,
                                                        linguistic_scale_size) == res_tuple, 'MTA operator is broken'


def generate_batch(numbers_quantity, batch_size, bits_per_number, bits_per_vector_for_inputs,
                   bits_per_vector_for_outputs, two_tuple_weight_precision,
                   two_tuple_alpha_precision,
                   two_tuple_largest_scale_size, mta_encoding, encoder=None, decoder=None):
    raw_dataset = []
    # structure: [
    #                  [ [ tuple1, tuple2], tuple_answer ],
    #                  [ [ tuple1, tuple2], tuple_answer ],
    # ]
    for batch_index in range(batch_size):
        # 1. generate the tuples
        tuples = []
        for _ in range(numbers_quantity):
            random_index = random.randint(0, two_tuple_largest_scale_size - 1)
            random_alpha = round(random.uniform(-.5, .5), two_tuple_alpha_precision)
            if mta_encoding in (MTATask.MTAEncodingType.full_no_weights, MTATask.MTAEncodingType.compact):
                weight = None
                first_tuple = tpr.model_2_tuple.Model2Tuple(term_index=random_index,
                                                            alpha=random_alpha,
                                                            linguistic_scale_size=two_tuple_largest_scale_size,
                                                            weight=weight)
            elif mta_encoding == MTATask.MTAEncodingType.full:
                weight = round(1 / numbers_quantity, two_tuple_weight_precision)
                first_tuple = tpr.model_2_tuple.Model2Tuple(term_index=random_index,
                                                            alpha=random_alpha,
                                                            linguistic_scale_size=two_tuple_largest_scale_size,
                                                            weight=weight)
            else:
                raise NotImplemented(f'Encoding scheme <{mta_encoding}> is not currently supported')

            tuples.append(first_tuple)

        # 2. aggregate the tuples
        mta_result_tuple = tpr.model_2_tuple.aggregate_model_tuples(tuples, two_tuple_largest_scale_size)

        raw_dataset.append([tuples, mta_result_tuple])

    example_input = new_empty_placeholder(numbers_quantity, batch_size, bits_per_number,
                                          bits_per_vector_for_inputs)
    example_output = np.zeros((batch_size, bits_per_number, bits_per_vector_for_outputs))

    if mta_encoding == MTATask.MTAEncodingType.full or mta_encoding == MTATask.MTAEncodingType.full_no_weights:
        example_input, example_output = pack_with_full_mta_encoding(raw_dataset, numbers_quantity, bits_per_number,
                                                                    two_tuple_largest_scale_size, example_input,
                                                                    example_output, encoder, decoder, mta_encoding)
    elif mta_encoding == MTATask.MTAEncodingType.compact:
        example_input, example_output = pack_with_compact_mta_encoding(raw_dataset, bits_per_number, example_input,
                                                                       example_output)
    else:
        raise NotImplemented(f'Encoding scheme <{mta_encoding}> is not currently supported')

    return example_input, example_output


def encode_compact_model_2_tuple(model_2_tuple):
    index, alpha, weight = tpr.model_2_tuple.FillerFactory._to_tpr_fillers(model_2_tuple)

    separator_index = len(index)
    return np.concatenate((index, np.array([0]), alpha)), separator_index


def pack_with_compact_mta_encoding(raw_dataset, bits_per_number, example_input, example_output):
    """"
    Pack scheme for the single entry:
    0 0 0 |^^^
    0 0 0 |-filler of index of tuple #1
    1 0 0 |vvv
    0 1 0 |-marker "end of filler"
    0 0 0 |^^^
    0 0 0 |-filler of alpha of tuple #1
    1 0 0 |vvv
    0 0 1 |-marker "end of tuple"
    0 0 0 |^^^
    0 0 0 |-filler of index of tuple #2
    1 0 0 |vvv
    0 1 0 |-marker "end of filler"
    0 0 0 |^^^
    0 0 0 |-filler of alpha of tuple #2
    1 0 0 |vvv
    0 0 1 |-marker "end of tuple"
    1 1 1 |-marker "end of expression" - added at the end of the pipeline
    """
    for sample_index, (tuples, mta_result_tuple) in enumerate(raw_dataset):
        for tuple_index, tuple_ith in enumerate(tuples):
            vec, sep_index = encode_compact_model_2_tuple(tuple_ith)
            assert len(vec) == bits_per_number, 'THEY SHOULD BE EQUAL'

            vec_starts_at = tuple_index * (bits_per_number + 1)
            vec_finishes_at = vec_starts_at + bits_per_number
            example_input[sample_index, vec_starts_at:vec_finishes_at, 0] = vec
            example_input[sample_index, vec_starts_at + sep_index, 1] = 1  # marker "end of filler"
            example_input[sample_index, vec_finishes_at, 2] = 1  # marker "end of tuple"

        vec, sep_index = encode_compact_model_2_tuple(mta_result_tuple)
        example_output[sample_index, :, 0] = vec
        example_output[sample_index, sep_index, 1] = 1

    return example_input, example_output


def pack_with_full_mta_encoding(raw_dataset, numbers_quantity, bits_per_number,
                                two_tuple_largest_scale_size, example_input, example_output, encoder, decoder,
                                mta_encoding):
    batch_tuples = []
    batch_result_tuples = []
    for tuples, mta_result_tuple in raw_dataset:
        # 3. encode the tuples
        encoded_tuples = [tpr.model_2_tuple.encode_model_2_tuple(i, encoder=encoder)[0] for i in tuples]
        flattened_encoded_tuples = [tpr.flattenize_per_tensor_representation(i) for i in encoded_tuples]
        encoded_mta_result_tuple = tpr.model_2_tuple.encode_model_2_tuple(mta_result_tuple, encoder=encoder)[0]
        flattened_encoded_mta_result_tuple = tpr.flattenize_per_tensor_representation(encoded_mta_result_tuple)
        print(f'TPR flattened shape is: {flattened_encoded_mta_result_tuple.shape}')

        batch_tuples.append(flattened_encoded_tuples)
        batch_result_tuples.append(flattened_encoded_mta_result_tuple)

    # 4. pack vectors in the training sample
    for sample_index, sample in enumerate(batch_tuples):
        for tpr_index, vec in enumerate(sample):
            number_starts_at = tpr_index * (bits_per_number + 1)
            number_finishes_at = number_starts_at + bits_per_number

            example_input[sample_index, number_starts_at:number_finishes_at, 0] = vec

            # binary operation encoding (0 1 0) in the original paper
            example_input[sample_index, number_finishes_at, 1] = 1

        example_output[sample_index, :, 0] = batch_result_tuples[sample_index]

    log_generated_mta_samples(example_input, example_output, bits_per_number, numbers_quantity,
                              two_tuple_largest_scale_size, is_verbose=True, encoder=encoder, decoder=decoder,
                              mta_encoding=mta_encoding)
    return example_input, example_output


def generate_batches(num_batches, batch_size, bits_per_vector, numbers_quantity,
                     two_tuple_weight_precision, two_tuple_alpha_precision, two_tuple_largest_scale_size,
                     mta_encoding):
    if mta_encoding == MTATask.MTAEncodingType.full:
        first_tuple = tpr.model_2_tuple.Model2Tuple(term_index=1, alpha=0.2,
                                                    linguistic_scale_size=two_tuple_largest_scale_size,
                                                    weight=.0)
        # 0. obtaining encoder and decoder networks for further re-use
        encoded_tuple, encoder = tpr.model_2_tuple.encode_model_2_tuple(first_tuple)
        flattened_encoded_tuple = tpr.flattenize_per_tensor_representation(encoded_tuple)
        decoded_tuple, decoder = tpr.model_2_tuple.decode_model_2_tuple_tpr(flattened_encoded_tuple)

        bits_per_number = single_tpr_len(two_tuple_largest_scale_size)
    elif mta_encoding == MTATask.MTAEncodingType.full_no_weights:
        first_tuple = tpr.model_2_tuple.Model2Tuple(term_index=1, alpha=0.2,
                                                    linguistic_scale_size=two_tuple_largest_scale_size,
                                                    weight=None)
        # 0. obtaining encoder and decoder networks for further re-use
        encoded_tuple, encoder = tpr.model_2_tuple.encode_model_2_tuple(first_tuple)
        flattened_encoded_tuple = tpr.flattenize_per_tensor_representation(encoded_tuple)
        decoded_tuple, decoder = tpr.model_2_tuple.decode_model_2_tuple_tpr(flattened_encoded_tuple,
                                                                            model_2_tuple_has_weights=False)

        bits_per_number = single_tpr_len(two_tuple_largest_scale_size)
    elif mta_encoding == MTATask.MTAEncodingType.compact:
        first_tuple = tpr.model_2_tuple.Model2Tuple(term_index=1, alpha=0.2,
                                                    linguistic_scale_size=two_tuple_largest_scale_size,
                                                    weight=.0)
        # 0. encoder and decoder are not needed in this case
        encoder = None
        decoder = None

        index, alpha, weight = tpr.model_2_tuple.FillerFactory._to_tpr_fillers(first_tuple)
        # meaning that every 2-tuple is encoded by filler or index, then marker, then filler of alpha
        bits_per_number = len(index) + 1 + len(alpha)
    else:
        raise ValueError(f'Encoding scheme <{mta_encoding}> is not currently supported')

    # logger.info(f'Bits per number in dataset entry: {bits_per_number}')

    batches = []
    for i in range(num_batches):
        # actually just a requirement of the network architecture
        # TODO: need to understand why exactly it requires such a blob
        bits_per_vector_for_inputs = bits_per_vector + 1

        bits_per_vector_for_outputs = bits_per_vector

        inputs, outputs = generate_batch(numbers_quantity,
                                         batch_size,
                                         bits_per_number,
                                         bits_per_vector_for_inputs,
                                         bits_per_vector_for_outputs,
                                         two_tuple_weight_precision,
                                         two_tuple_alpha_precision,
                                         two_tuple_largest_scale_size,
                                         mta_encoding,
                                         encoder=encoder,
                                         decoder=decoder)

        # TODO: should it be a full row of ones as it is in other tasks? Or as in
        # TODO: binary arithmetic paper - just a flag?
        eos = np.ones([batch_size, 1, bits_per_vector_for_inputs])
        output_inputs = np.zeros((batch_size, bits_per_number, bits_per_vector_for_inputs))

        full_inputs = np.concatenate((inputs[:, :-1, :], eos, output_inputs), axis=1)

        batches.append(
            (
                bits_per_number,
                full_inputs,
                outputs
            )
        )
    return batches


def load_batches_from_file(path: Path):
    with open(path, 'rb') as output:
        return pickle.load(output)


class MTATaskData(BinaryVectorErrorEstimator):
    def generate_batches(self, num_batches, batch_size, bits_per_vector=3, curriculum_point=20, max_seq_len=4,
                         curriculum='uniform', pad_to_max_seq_len=False, numbers_quantity=3,
                         two_tuple_weight_precision=1, two_tuple_alpha_precision=1, two_tuple_largest_scale_size=5,
                         cli_mode=False, mta_encoding=MTATask.MTAEncodingType.full):
        if curriculum != 'none':
            sys.exit(f'Current "{curriculum}" curriculum is not supported by AverageSumTaskData task')

        arguments = {
            'num_batches': num_batches,
            'batch_size': batch_size,
            'bits_per_vector': bits_per_vector,
            'numbers_quantity': numbers_quantity,
            'two_tuple_weight_precision': two_tuple_weight_precision,
            'two_tuple_alpha_precision': two_tuple_alpha_precision,
            'two_tuple_largest_scale_size': two_tuple_largest_scale_size,
            'mta_encoding': mta_encoding,
        }

        if cli_mode:
            kwargs = {f'--{key}': value for key, value in arguments.items()}
            batch_path = TMP_ARTIFACTS_PATH / 'batch.pkl'
            kwargs['--serialized_path'] = str(batch_path)
            kwargs['--mode'] = 'generate'
            cli_generator_path = Path(__file__).parent / 'generator_cli.py'
            res_output = run_console_tool(tool_path=cli_generator_path, **kwargs)
            # print(f'SUBPROCESS: {str(res_output.stdout.decode("utf-8"))}')
            # print(f'SUBPROCESS: {str(res_output.stderr.decode("utf-8"))}')
            return load_batches_from_file(batch_path)

        return generate_batches(**arguments)


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    generator = MTATaskData()
    before = time.time()
    data = generator.generate_batches(num_batches=10, batch_size=32, curriculum='none',
                                      numbers_quantity=2,
                                      two_tuple_weight_precision=1,
                                      two_tuple_alpha_precision=1,
                                      two_tuple_largest_scale_size=5,
                                      cli_mode=False)
    after = time.time() - before
    print(f'Time elapsed: {after}')
