import argparse
import sys
from pathlib import Path

import numpy as np

from infer import infer_model
from tasks.operators.mta.generator import MTATaskData, pack_with_compact_mta_encoding, new_empty_placeholder, \
    encode_compact_model_2_tuple
from tasks.operators.mta.task import MTATask
from tasks.operators.tpr_toolkit.core.model_2_tuple import Model2Tuple, aggregate_model_tuples, FillerFactory


def _generate_data(mta_encoding, num_experts, scale_size):
    generator_args = dict(
        num_batches=1,
        batch_size=32,
        bits_per_vector=3,
        curriculum_point=None,
        max_seq_len=-1,  # made intentionally, generator will define TPR length itself
        curriculum='none',
        pad_to_max_seq_len=False
    )

    generator_args['cli_mode'] = mta_encoding in (
        MTATask.MTAEncodingType.full, MTATask.MTAEncodingType.full_no_weights)
    generator_args['numbers_quantity'] = num_experts
    generator_args['two_tuple_weight_precision'] = 1
    generator_args['two_tuple_alpha_precision'] = 1
    generator_args['two_tuple_largest_scale_size'] = scale_size
    generator_args['mta_encoding'] = mta_encoding

    data_generator = MTATaskData()

    return data_generator.generate_batches(**generator_args)[0], data_generator


def test_model(directory_path: Path, mta_encoding, num_experts, scale_size):
    (seq_len, inputs, labels), data_generator = _generate_data(mta_encoding, num_experts, scale_size)

    outputs = infer_model(directory_path, inputs=inputs, seq_len=seq_len)

    error = data_generator.error_per_seq(labels, outputs, 32)

    return error


def demo_summator(directory_path: Path, numbers, mta_encoding, num_experts, scale_size):
    (seq_len, inputs, labels), data_generator = _generate_data(mta_encoding, num_experts, scale_size)

    bits_per_vector = 3
    bits_per_vector_for_outputs = bits_per_vector

    example_output = np.zeros((1, seq_len, bits_per_vector_for_outputs))

    raw_dataset = [
        [numbers, aggregate_model_tuples(numbers, scale_size)]
    ]

    example_input, _ = pack_with_compact_mta_encoding(raw_dataset, seq_len, inputs, example_output)

    outputs = infer_model(directory_path, inputs=inputs, seq_len=seq_len)

    _, separator_index = encode_compact_model_2_tuple(numbers[0])

    predicted_raw = outputs[0][:, 0]
    term_filler = predicted_raw[:separator_index]
    alpha_filler = predicted_raw[separator_index + 1:]

    term_index, alpha, _ = FillerFactory.decode_fillers(term_filler, alpha_filler, None)
    result = Model2Tuple(term_index=term_index, alpha=alpha, linguistic_scale_size=5, weight=None)
    return result


if __name__ == '__main__':
    LINGUISTIC_SCALE_SIZE = 5

    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_model_filename', default='results/frozen_model.pb', type=str,
                        help='Frozen model file to import')
    parser.add_argument('--num_experts', required=False, type=int,
                        help='Optional. Needed for average sum task and stands for the quantity of numbers to be used'
                             'for calculations')
    parser.add_argument('--mta_encoding', choices=(
        MTATask.MTAEncodingType.full,
        MTATask.MTAEncodingType.compact,
        MTATask.MTAEncodingType.full_no_weights),
                        required=False, default=MTATask.MTAEncodingType.full,
                        help='Optional. Specifies how dataset is encoded. Full means 2-tuple is fed to network'
                             'as full TPR. Compact means 2-tuple is fed to network as two fillers: term and projection')

    args = parser.parse_args()

    model = Path(args.frozen_model_filename)

    overall_err = test_model(model.parent, mta_encoding=args.mta_encoding, num_experts=args.num_experts,
                             scale_size=LINGUISTIC_SCALE_SIZE)
    print(f'Overall quality of model. Error: {overall_err}')

    tuples = []
    if args.num_experts == 2:
        tuples.append(Model2Tuple(term_index=3, alpha=0.2, linguistic_scale_size=LINGUISTIC_SCALE_SIZE, weight=None))
        tuples.append(Model2Tuple(term_index=2, alpha=0, linguistic_scale_size=LINGUISTIC_SCALE_SIZE, weight=None))
    else:
        sys.exit(f'Provide {args.num_experts} examples of 2-tuples to test the model')

    demo_result = demo_summator(model.parent, numbers=tuples, mta_encoding=args.mta_encoding,
                                num_experts=args.num_experts, scale_size=LINGUISTIC_SCALE_SIZE)
    summands_str = ', '.join([str(i) for i in tuples])

    expected_res = Model2Tuple(term_index=3, alpha=-0.4, linguistic_scale_size=5, weight=None)

    print(f"MTA({summands_str}) ~= {demo_result}. Expected: {expected_res}")
