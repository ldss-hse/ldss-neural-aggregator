import enum
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tap import Tap

from infer import prepare_graph_for_inference
from infer_mta import _generate_data as generate_tpr_data
from infer import _generate_data as generate_binary_data


class AnalyzerCLIArgumentParser(Tap):
    device: str


class DeviceType(str, enum.Enum):
    CPU = '/cpu:0'
    GPU = '/gpu:0'
    TPU = '/tpu:0'

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, device):
        if 'cpu' in device:
            return DeviceType.CPU
        if 'gpu' in device:
            return DeviceType.GPU
        return DeviceType.TPU


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
            bits_per_number: int = None, device_type: DeviceType = DeviceType.CPU):
    run_metadata = None
    melting_rounds = 1
    for i in range(melting_rounds):
        print(f'\t [{i + 1}/{melting_rounds}] Inference on {device_type} started...')
        run_metadata = tf.compat.v1.RunMetadata()
        if encoding is EncodingType.TPR:
            (seq_len, inputs, labels), data_generator = generate_tpr_data(str(mta_encoding), num_experts, scale_size)
        else:
            (seq_len, inputs, labels), data_generator = generate_binary_data(bits_per_number, num_experts)

        with tf.compat.v1.device(str(device_type)):
            graph, (inputs_placeholder, seq_len_placeholder), y = prepare_graph_for_inference(frozen_path,
                                                                                              graph_file_name='frozen_graph_no_device.pb',
                                                                                              prefix='prefix')
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
        print(f'Device: {device.device} Ops count: {len(device.node_stats)}')

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


def remove_device_from_graph(frozen_path: Path, source_name: str, final_name: str):
    graph, _, y = prepare_graph_for_inference(frozen_path, graph_file_name=source_name)
    with graph.as_default():
        graph_def = graph.as_graph_def()
        for node in graph_def.node:
            node.device = ""

        tf.compat.v1.train.write_graph(graph_def, str(frozen_path), final_name, False)


def prepare_report(all_rows: list, report_path: Path):
    print('Preparing report...')
    df = pd.DataFrame(all_rows)
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df[cols]

    df.to_csv(report_path, sep='\t')


def main(args: AnalyzerCLIArgumentParser):
    all_rows = []
    models_count = len(FROZEN_MODEL_MAPPING)
    device_type = DeviceType.from_str(args.device)
    for model_idx, (model_id, model_info) in enumerate(FROZEN_MODEL_MAPPING.items()):
        print(f'[{model_idx + 1}/{models_count}] Running inference for <{model_id}>...')

        model_dir_name = model_info['model_dir_name']
        frozen_dir_path = Path(__file__).parent / 'trained_models' / model_dir_name

        no_device_name = 'frozen_graph_no_device.pb'
        if not (frozen_dir_path / no_device_name).exists():
            original_name = 'frozen_graph.pb'
            remove_device_from_graph(frozen_dir_path, original_name, no_device_name)

        res = analyze(frozen_path=frozen_dir_path,
                      encoding=model_info['encoding'],
                      mta_encoding=model_info.get('mta_encoding'),
                      num_experts=model_info.get('num_experts'),
                      scale_size=model_info.get('scale_size'),
                      bits_per_number=model_info.get('bits_per_number'),
                      device_type=device_type)
        res['name'] = model_id
        res['device_type'] = str(device_type)
        all_rows.append(res)

    report_path = Path(__file__).parent / 'artifacts' / 'report.tsv'
    prepare_report(all_rows, report_path)


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.compat.v1.disable_eager_execution()

    args = AnalyzerCLIArgumentParser().parse_args()

    main(args)
