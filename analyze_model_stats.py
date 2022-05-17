import enum
from pathlib import Path

import pandas as pd
import tensorflow as tf

from infer import prepare_graph_for_inference
from infer_mta import _generate_data as generate_tpr_data
from infer import _generate_data as generate_binary_data


class EncodingType(str, enum.Enum):
    TPR = 'TPR'
    BINARY = 'BINARY'


class TPREncodingStrategy(str, enum.Enum):
    COMPACT = 'compact'
    FULL_NO_WEIGHTS = 'full_no_weights'
    FULL = 'full'

    def __str__(self):
        return self.value


FROZEN_MODEL_MAPPING = {
    'TPR (17 bits, 2 experts)': {
        'encoding': EncodingType.TPR,
        'num_experts': 2,
        'scale_size': 5,
        'mta_encoding': TPREncodingStrategy.COMPACT,
        'model_dir_name': 'mta_v1/17_bits_256_memory_2_experts_local_compact_binary_encoding_binary_layout'
    },
    'TPR (48 bits, 2 experts)': {
        'encoding': EncodingType.TPR,
        'num_experts': 2,
        'scale_size': 5,
        'mta_encoding': TPREncodingStrategy.FULL_NO_WEIGHTS,
        'model_dir_name': 'mta_v1/48_bits_256_memory_2_experts_local_full_no_weights_binary_encoding_binary_layout'
    },
    'TPR (104 bits, 2 experts)': {
        'encoding': EncodingType.TPR,
        'num_experts': 2,
        'scale_size': 5,
        'mta_encoding': TPREncodingStrategy.FULL,
        'model_dir_name': 'mta_v1/104_bits_256_memory_2_experts_local_full_non_binary_encoding_binary_layout'
    },
    'Binary (6 bits, 2 experts)': {
        'encoding': EncodingType.BINARY,
        'bits_per_number': 6,
        'num_experts': 2,
        'model_dir_name': 'average_binary_sum_v3/4_bits_128_memory_2_experts_contrib'
    },
    'Binary (8 bits, 2 experts)': {
        'encoding': EncodingType.BINARY,
        'bits_per_number': 8,
        'num_experts': 2,
        'model_dir_name': 'average_binary_sum_v3/8_bits_256_memory_2_experts_contrib'
    },
    'Binary (10 bits, 2 experts)': {
        'encoding': EncodingType.BINARY,
        'bits_per_number': 10,
        'num_experts': 2,
        'model_dir_name': 'average_binary_sum_v3/10_bits_256_memory_2_experts_contrib'
    },
    'Binary (16 bits, 2 experts)': {
        'encoding': EncodingType.BINARY,
        'bits_per_number': 16,
        'num_experts': 2,
        'model_dir_name': 'average_binary_sum_v3/16_bits_256_memory_2_experts_contrib'
    },
    'Binary (6 bits, 3 experts)': {
        'encoding': EncodingType.BINARY,
        'bits_per_number': 6,
        'num_experts': 3,
        'model_dir_name': 'average_binary_sum_v2/6_bits_256_memory_3_experts_contrib'
    },
    'Binary (8 bits, 3 experts)': {
        'encoding': EncodingType.BINARY,
        'bits_per_number': 8,
        'num_experts': 3,
        'model_dir_name': 'average_binary_sum_v2/8_bits_256_memory_3_experts_local'
    },
    'Binary (8 bits, 512 locations, 3 experts)': {
        'encoding': EncodingType.BINARY,
        'bits_per_number': 8,
        'num_experts': 3,
        'model_dir_name': 'average_binary_sum_v2/8_bits_512_memory_3_experts_contrib'
    }
}


def analyze(frozen_path: Path, mta_encoding: TPREncodingStrategy, num_experts: int, scale_size: int,
            encoding: EncodingType,
            bits_per_number: int = None, is_gpu: bool = False):

    # this report does not work probably as we did not use during training phase
    # opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    # trainable_params = tf.compat.v1.profiler.profile(graph, options=opts, cmd='op')
    # print('Trainable parameters = ', trainable_params.total_parameters)

    # this report does not work as it has several shapes undefined
    # how to reshape the graph - open question
    # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # flops = tf.compat.v1.profiler.profile(graph, options=opts, cmd='op')
    # print('Theoretical FLOPs = ', flops.total_float_ops)

    run_metadata = None
    melting_rounds = 1
    for i in range(melting_rounds):
        print(f'\t [{i + 1}/{melting_rounds}] Inference started...')
        run_metadata = tf.compat.v1.RunMetadata()
        if encoding is EncodingType.TPR:
            (seq_len, inputs, labels), data_generator = generate_tpr_data(str(mta_encoding), num_experts, scale_size)
        else:
            (seq_len, inputs, labels), data_generator = generate_binary_data(bits_per_number, num_experts)

        if is_gpu:
            device_name = "/gpu:0"
        else:
            device_name = "/gpu:0"
        with tf.device(device_name):
            graph, (inputs_placeholder, seq_len_placeholder), y = prepare_graph_for_inference(frozen_path)
            with tf.compat.v1.Session(graph=graph,
                                      config=tf.compat.v1.ConfigProto(allow_soft_placement=False,
                                                                      log_device_placement=False)) as sess:
                _ = sess.run(y,
                             feed_dict={
                                 inputs_placeholder: inputs,
                                 seq_len_placeholder: seq_len
                             },
                             options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                             run_metadata=run_metadata
                             )
    for device in run_metadata.step_stats.dev_stats:
        device_name = device.device
        # if not (device_name.lower().endswith("cpu:0") or device_name.lower().endswith("gpu:0")):
        #     continue
        print(f'Device: {device.device} Ops count: {len(device.node_stats)}')
        # for node in device.node_stats:
        #     print("!!!   ", node.node_name)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory()
    time_memory = tf.compat.v1.profiler.profile(graph, options=opts, cmd='op', run_meta=run_metadata)

    batch_size = 32
    return {
        'Total exec time (ms)': time_memory.total_exec_micros / 1000,  # as it is in microseconds
        'CPU exec time (ms)': time_memory.total_cpu_exec_micros / 1000,  # as it is in microseconds
        'Requested bytes (MB)': time_memory.total_requested_bytes / (batch_size * 1000 ** 2),  # as it is in bytes
        'Output bytes (MB)': time_memory.total_output_bytes / (batch_size * 1000 ** 2),  # as it is in bytes
        'GFLOPs': time_memory.total_float_ops / (1000 ** 3),  # as we want to have it as multiplier of 1 * 10^9
    }


def prepare_report(all_rows: list, report_path: Path):
    print('Preparing report...')
    df = pd.DataFrame(all_rows)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df.to_csv(report_path, sep='\t')

    pd.set_option('display.max_columns', None)
    print(df)


def main():
    all_rows = []
    models_count = len(FROZEN_MODEL_MAPPING)
    for model_idx, (model_id, model_info) in enumerate(FROZEN_MODEL_MAPPING.items()):
        print(f'[{model_idx + 1}/{models_count}] Running inference for <{model_id}>...')
        model_dir_name = model_info['model_dir_name']

        frozen_dir_path = Path(__file__).parent / 'trained_models' / model_dir_name

        res = analyze(frozen_path=frozen_dir_path,
                      encoding=model_info['encoding'],
                      mta_encoding=model_info.get('mta_encoding'),
                      num_experts=model_info.get('num_experts'),
                      scale_size=model_info.get('scale_size'),
                      bits_per_number=model_info.get('bits_per_number'),
                      is_gpu=True)
        res['name'] = model_id
        all_rows.append(res)
        break

    report_path = Path(__file__).parent / 'artifacts' / 'report.tsv'
    prepare_report(all_rows, report_path)


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.compat.v1.disable_eager_execution()

    main()
