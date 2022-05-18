from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_report(report_path: Path):
    df = pd.read_csv(report_path, sep='\t')
    df.set_index('name', inplace=True)
    df.drop(['Unnamed: 0', 'device_type'], axis=1, inplace=True)

    rename_mapping = {
        'Total exec time (ms)': 'total_time',
        'CPU exec time (ms)': 'cpu_time',
        'Requested bytes (MB)': 'requested_bytes',
        'Output bytes (MB)': 'output_bytes',
        'GFLOPs': 'gflops'
    }
    df.rename(columns=rename_mapping, inplace=True)
    return df


def main():
    reports_path = Path(__file__).parent / 'trained_models' / 'stats_reports'
    cpu_df = load_report(reports_path / 'report_CPU.tsv')
    gpu_df = load_report(reports_path / 'report_GPU.tsv')

    joint_df = cpu_df.join(gpu_df, lsuffix='_CPU', rsuffix='_GPU')

    # although gflops vary across devices, it was decided to use only CPU report
    # as it is more trustworthy
    joint_df.drop('gflops_GPU', axis=1, inplace=True)

    # to not confuse, report accelerator time as anyway in any mode a model is executed
    # in a hetero mode
    joint_df.drop('cpu_time_CPU', axis=1, inplace=True)
    joint_df.drop('cpu_time_GPU', axis=1, inplace=True)

    joint_df.sort_index(axis=1, inplace=True)

    device_one_gflop_to_watt = {
        'cpu': 12,
        'gpu': 12,
        'ncs2': (10 ** (-12)) * (10 ** 9)
    }
    joint_df['ncs2_W'] = joint_df['gflops_CPU'] * device_one_gflop_to_watt['ncs2']

    joint_df['gflops_CPU'] = joint_df['gflops_CPU'].round(2)
    joint_df['output_bytes_CPU'] = joint_df['output_bytes_CPU'].round(2)
    joint_df['output_bytes_GPU'] = joint_df['output_bytes_GPU'].round(2)
    joint_df['requested_bytes_CPU'] = joint_df['requested_bytes_CPU'].round(2)
    joint_df['requested_bytes_GPU'] = joint_df['requested_bytes_GPU'].round(2)
    joint_df['total_time_CPU'] = joint_df['total_time_CPU'].round(2)
    joint_df['total_time_GPU'] = joint_df['total_time_GPU'].round(2)
    joint_df['ncs2_W'] = joint_df['ncs2_W'].round(4)

    joint_df.to_csv(reports_path / 'accumulated.tsv', sep='\t')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=joint_df.values, colLabels=joint_df.columns, loc='center')

    pp = PdfPages(reports_path / "table.pdf")
    pp.savefig(fig, bbox_inches='tight')
    pp.close()


if __name__ == '__main__':
    main()
