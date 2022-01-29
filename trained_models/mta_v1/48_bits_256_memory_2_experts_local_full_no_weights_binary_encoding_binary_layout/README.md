# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-09-09|
|Task| mta (full binary layout, full no weights  binary encoding)|
|Error per sequence (start)| 2.0 |
|Loss function value (start)| 7.8374675750732425 |
|Error per sequence| 0.0 |
|Loss function value| 0.001199930958682671 |
|Training iterations| 830000 |
|Additional parameters| `mta_encoding=full_no_weights,num_experts=2` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                    --num_train_steps 1000000 --steps_per_eval 1000 \
                    --use_local_impl yes --curriculum none --device cpu --num_bits_per_vector 3 \
                    --num_memory_locations 256 --max_seq_len 4 \
                    --task mta --mta_encoding full_no_weights --num_experts 2
```

## Inference command

**N/A**

## Logs

[Logs](./out.log)
