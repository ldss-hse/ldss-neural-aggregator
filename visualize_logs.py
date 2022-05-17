import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import TMP_ARTIFACTS_PATH


def get_i8n_name(name, name_type, lang):
    words = {
        'error': {
            'y_title': {
                'en': 'error',
                'ru': 'Количество ошибочных бит',
            },
            'chart_title': {
                'en': 'Error per sequence',
                'ru': 'Количество ошибочных бит в предсказаниях',
            },
        },
        'loss': {
            'y_title': {
                'en': 'loss',
                'ru': 'Потери',
            },
            'chart_title': {
                'en': 'Total loss',
                'ru': 'Значение функции потерь',
            },
        },
        'common': {
            'title': {
                'en': 'NTM training dynamics for MTA \noperator for two assessments',
                'ru': 'Динамика обучения Нейронной Машины Тьюринга \nдля агрегации двух оценок\n',
            },
            'training_steps': {
                'en': 'training steps',
                'ru': 'Итерации обучения',
            },
        },
        'encoding': {
            'compact': {
                'en': 'compact encoding',
                'ru': 'компактное кодирование',
            },
            'full_no_weights': {
                'en': 'full encoding',
                'ru': 'полное кодирование',
            },
            '4 bits, 128 memory': {
                'en': '4 bits, 128 memory',
                'ru': '4 бит, 128 ячеек памяти',
            },
            '6 bits, 256 memory': {
                'en': '4 bits, 128 memory',
                'ru': '6 бит, 256 ячеек памяти',
            },
            '8 bits, 256 memory': {
                'en': '8 bits, 128 memory',
                'ru': '8 бит, 256 ячеек памяти',
            },
            '8 bits, 512 memory': {
                'en': '8 bits, 512 memory',
                'ru': '8 бит, 512 ячеек памяти',
            },
            '10 bits, 256 memory': {
                'en': '10 bits, 128 memory',
                'ru': '10 бит, 256 ячеек памяти',
            },
            '16 bits, 256 memory': {
                'en': '16 bits, 128 memory',
                'ru': '16 бит, 256 ячеек памяти',
            },
            '48 bits, 256 memory': {
                'en': '48 bits, 128 memory',
                'ru': '48 бит, 256 ячеек памяти',
            },
            '17 bits, 256 memory': {
                'en': '17 bits, 128 memory',
                'ru': '17 бит, 256 ячеек памяти',
            }
        }
    }
    return words[name_type][name][lang]


def create_chart(ticks, granularity=1000, type='error', bits_per_number=4, series={}, config=None):
    if config is None:
        lang = 'en'
    else:
        lang = config['lang']
    font_settings = {
        'fontname': 'Times New Roman',
        'fontsize': 12
    }

    if type == 'error':
        y_title = get_i8n_name(name='y_title', name_type='error', lang=lang)
        chart_title = get_i8n_name(name='chart_title', name_type='error', lang=lang)
    else:  # it is loss
        y_title = get_i8n_name(name='y_title', name_type='loss', lang=lang)
        chart_title = get_i8n_name(name='chart_title', name_type='loss', lang=lang)

    new_ticks = [i for i in ticks if i % granularity == 0]
    filtered_data_df = pd.DataFrame({
        'x': ticks,
        **series
    })
    filtered_data_df = filtered_data_df[filtered_data_df['x'] <= new_ticks[-1]]
    plt.close()

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'

    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                  CB91_Purple, CB91_Violet]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    for key in series:
        if config is None or config.get('labels') is None:
            plt.plot('x', key, data=filtered_data_df, label=key)
        else:
            # plt.plot('x', key, data=filtered_data_df, label=config.get('labels')[key], linestyle='dashed')
            plt.plot('x', key, data=filtered_data_df, label=config.get('labels')[key])

    plt.ylabel(y_title, **font_settings)
    plt.xlabel(get_i8n_name(name="training_steps", name_type="common", lang=lang), **font_settings)
    plt.locator_params(nbins=5)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # plt.xticks(new_ticks, **font_settings)
    plt.yticks(**font_settings)

    if lang not in ('ru', 'en'):
        plt.title(f'{get_i8n_name(name="title", name_type="common", lang=lang)}, {chart_title}')
    plt.legend()
    return plt


def save_chart(plot, type, bits_per_number, file_name=None, file_format='svg', language='ru'):
    try:
        TMP_ARTIFACTS_PATH.mkdir()
    except FileExistsError:
        pass

    if file_name is None:
        path_template = TMP_ARTIFACTS_PATH / f'{bits_per_number}_bit_{type}_{language}.{file_format}'
    else:
        path_template = TMP_ARTIFACTS_PATH / f'{file_name}_{language}.{file_format}'

    plot.savefig(path_template, format=file_format, dpi=600)


PARSABLE_PATTERN = re.compile(r'''
                        (.*EVAL_PARSABLE:\s)
                        (?P<step>\d+),
                        (?P<error>\d+\.\d+(e[+-]\d+)?),
                        (?P<loss>\d+\.\d+(e[+-]\d+)?).*''', re.VERBOSE)


def get_history(log_path):
    steps = []
    errors = []
    losses = []
    with open(log_path) as f:
        for line in f:
            matched = re.match(PARSABLE_PATTERN, line)
            if not matched:
                continue
            steps.append(int(matched.group('step')))
            errors.append(float(matched.group('error')))
            losses.append(float(matched.group('loss')))
    return steps, errors, losses


def main(args):
    steps, errors, losses = get_history(args.log_path)
    plot = create_chart(series={'Error': errors}, ticks=steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number)
    plot = create_chart(series={'Loss': losses}, ticks=steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number)


def main_mta_all(args):
    mem_128_4_bit_steps, mem_128_4_bit_errors, mem_128_4_bit_losses = get_history(
        './trained_models/average_binary_sum_v1/out.log')
    mem_128_4_bit_steps = [*mem_128_4_bit_steps[9::10], mem_128_4_bit_steps[-1]]
    mem_128_4_bit_errors = [*mem_128_4_bit_errors[9::10], mem_128_4_bit_errors[-1]]
    mem_128_4_bit_losses = [*mem_128_4_bit_losses[9::10], mem_128_4_bit_losses[-1]]
    mem_256_6_bit_steps, mem_256_6_bit_errors, mem_256_6_bit_losses = get_history(
        './trained_models/average_binary_sum_v2/6_bits_256_memory_3_experts_contrib/out.log')
    mem_256_8_bit_steps, mem_256_8_bit_errors, mem_256_8_bit_losses = get_history(
        './trained_models/average_binary_sum_v2/8_bits_256_memory_3_experts_local/out.log')
    mem_512_8_bit_steps, mem_512_8_bit_errors, mem_512_8_bit_losses = get_history(
        './trained_models/average_binary_sum_v2/8_bits_512_memory_3_experts_contrib/out.log')
    mem_256_10_bit_steps, mem_256_10_bit_errors, mem_256_10_bit_losses = get_history(
        './trained_models/average_binary_sum_v2/10_bits_256_memory_3_experts_contrib/out.log')

    lang = 'ru'
    labels = {
        '4 bits, 128 memory': get_i8n_name(name='4 bits, 128 memory', name_type='encoding', lang=lang),
        '6 bits, 256 memory': get_i8n_name(name='6 bits, 256 memory', name_type='encoding', lang=lang),
        '8 bits, 256 memory': get_i8n_name(name='8 bits, 256 memory', name_type='encoding', lang=lang),
        '8 bits, 512 memory': get_i8n_name(name='8 bits, 512 memory', name_type='encoding', lang=lang),
        '10 bits, 256 memory': get_i8n_name(name='10 bits, 256 memory', name_type='encoding', lang=lang),
    }
    config = dict(lang=lang, labels=labels)

    series_dict = {
        '4 bits, 128 memory': mem_128_4_bit_errors,
        '6 bits, 256 memory': mem_256_6_bit_errors,
        '8 bits, 256 memory': mem_256_8_bit_errors,
        '8 bits, 512 memory': mem_512_8_bit_errors,
        '10 bits, 256 memory': mem_256_10_bit_errors,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_8_bit_steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number, config=config)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number, file_name='fig_4_c_error', file_format='pdf')

    series_dict = {
        '4 bits, 128 memory': mem_128_4_bit_losses,
        '6 bits, 256 memory': mem_256_6_bit_losses,
        '8 bits, 256 memory': mem_256_8_bit_losses,
        '8 bits, 512 memory': mem_512_8_bit_losses,
        '10 bits, 256 memory': mem_256_10_bit_losses,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_8_bit_steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number, config=config)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number, file_name='fig_4_d_loss', file_format='pdf')


def main_mta_2_experts_all(args):
    mem_128_4_bit_steps, mem_128_4_bit_errors, mem_128_4_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/4_bits_128_memory_2_experts_contrib/out.log')
    mem_256_6_bit_steps, mem_256_6_bit_errors, mem_256_6_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/6_bits_256_memory_2_experts_contrib/out.log')
    mem_256_8_bit_steps, mem_256_8_bit_errors, mem_256_8_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/8_bits_256_memory_2_experts_contrib/out.log')
    mem_256_10_bit_steps, mem_256_10_bit_errors, mem_256_10_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/10_bits_256_memory_2_experts_contrib/out.log')
    mem_256_16_bit_steps, mem_256_16_bit_errors, mem_256_16_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/16_bits_256_memory_2_experts_contrib/out.log')
    mem_256_16_bit_steps = [*mem_256_16_bit_steps[:12]]
    mem_256_16_bit_errors = [*mem_256_16_bit_errors[:12]]
    mem_256_16_bit_losses = [*mem_256_16_bit_losses[:12]]

    lang = 'ru'
    labels = {
        '6 bits, 256 memory': get_i8n_name(name='6 bits, 256 memory', name_type='encoding', lang=lang),
        '8 bits, 256 memory': get_i8n_name(name='8 bits, 256 memory', name_type='encoding', lang=lang),
        '10 bits, 256 memory': get_i8n_name(name='10 bits, 256 memory', name_type='encoding', lang=lang),
        '16 bits, 256 memory': get_i8n_name(name='16 bits, 256 memory', name_type='encoding', lang=lang),
    }
    config = dict(lang=lang, labels=labels)

    series_dict = {
        '6 bits, 256 memory': mem_256_6_bit_errors,
        '8 bits, 256 memory': mem_256_8_bit_errors,
        '10 bits, 256 memory': mem_256_10_bit_errors,
        '16 bits, 256 memory': mem_256_16_bit_errors,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_16_bit_steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number, config=config)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number, file_name='fig_4_a_error', file_format='pdf')

    series_dict = {
        '6 bits, 256 memory': mem_256_6_bit_losses,
        '8 bits, 256 memory': mem_256_8_bit_losses,
        '10 bits, 256 memory': mem_256_10_bit_losses,
        '16 bits, 256 memory': mem_256_16_bit_losses,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_16_bit_steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number, config=config)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number, file_name='fig_4_b_loss', file_format='pdf')


def main_mta_2_tuple_2_experts_v1_all(args):
    mem_256_104_bit_steps, mem_256_104_bit_errors, mem_256_104_bit_losses = get_history(
        './trained_models/mta_v1/104_bits_256_memory_2_experts_local_full_binary_layout/out.log')

    series_dict = {
        '104 bits, 256 memory': mem_256_104_bit_errors,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_104_bit_steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number)

    series_dict = {
        '104 bits, 256 memory': mem_256_104_bit_losses,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_104_bit_steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number)


def main_mta_2_tuple_2_experts_v2_all(args):
    mem_256_48_bit_steps, mem_256_48_bit_errors, mem_256_48_bit_losses = get_history(
        './trained_models/mta_v1/48_bits_256_memory_2_experts_local_full_no_weights_binary_encoding_binary_layout/out_till_283000.log')
    mem_256_17_bit_steps, mem_256_17_bit_errors, mem_256_17_bit_losses = get_history(
        './trained_models/mta_v1/17_bits_256_memory_2_experts_local_compact_binary_encoding_binary_layout/out_213000.log')

    labels = {
        '48 bits, 256 memory': get_i8n_name(name='48 bits, 256 memory', name_type='encoding', lang=args.language),
        '17 bits, 256 memory': get_i8n_name(name='17 bits, 256 memory', name_type='encoding', lang=args.language),
    }
    config = dict(lang=args.language, labels=labels)

    series_dict = {
        '48 bits, 256 memory': mem_256_48_bit_errors,
        '17 bits, 256 memory': mem_256_17_bit_errors,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])

    all_steps = (mem_256_48_bit_steps, mem_256_17_bit_steps)
    max_steps = all_steps[max(enumerate(all_steps), key=lambda x: len(x[1]))[0]]
    plot = create_chart(series=series_dict, ticks=max_steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number, config=config)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number, file_name='fig_6_a_error',
               file_format='pdf', language=args.language)

    series_dict = {
        '48 bits, 256 memory': mem_256_48_bit_losses,
        '17 bits, 256 memory': mem_256_17_bit_losses,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=max_steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number, config=config)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number, file_name='fig_6_b_loss',
               file_format='pdf', language=args.language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', required=True, type=str,
                        help='Path to training log')
    parser.add_argument('--granularity', required=False, type=int, default=5000,
                        help='Granularity for ticks')
    parser.add_argument('--bits_per_number', required=True, type=int,
                        help='Bits per number')
    parser.add_argument('--language', required=False, choices=('ru', 'en'), default='ru',
                        help='Bits per number')
    args = parser.parse_args()
    if args.log_path == 'mta_all':
        main_mta_all(args)
        main_mta_2_experts_all(args)
        # main_mta_2_tuple_2_experts_v1_all(args)
        main_mta_2_tuple_2_experts_v2_all(args)
    else:
        main(args)
