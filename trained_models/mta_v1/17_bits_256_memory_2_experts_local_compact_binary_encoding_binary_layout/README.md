# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-07-20|
|Task| mta (full binary layout, compact binary encoding)|
|Error per sequence (start)| 2.35 |
|Loss function value (start)| 6.518269777297974 |
|Error per sequence| 0.0 |
|Loss function value| 0.00021232106992101761 |
|Training iterations| 395000 |
|Additional parameters| `mta_encoding=compact,num_experts=2` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                    --num_train_steps 1000000 --steps_per_eval 1000 \
                    --use_local_impl yes --curriculum none --device cpu --num_bits_per_vector 3 \
                    --num_memory_locations 256 --max_seq_len 4 \
                    --task mta --mta_encoding compact --num_experts 2
```

## Inference command

```bash
python infer_mta.py --frozen_model_filename .\trained_models\mta_v1\17_bits_256_memory_2_experts_local_compact_binary_encoding_binary_layout\frozen_graph.pb \
                    --mta_encoding compact \
                    --num_experts 2
```

## Logs

[Logs](./out.log)
