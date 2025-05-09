# mopt_cloth
## Offline training
```python
python train_offline.py --task cloth-flatten --num_epoch 60 --out_dir exp/policy --data_file <DATA_FILE> --run_group offline
```

## Online fine-tuning

```python
python -m softgym.train_online --env_name ClothFlatten --task cloth-flatten --num_online 200 --nb_batches 4 --total_iter 300 --headless 1 --replay_buffer_size 100000 --run_group online_ft --batch_size 8192 --load_model <MODEL_FOLDER> --eps_pick 15e-2 --eps_place 15e-2 --replay_buffer_fill <FILL_FILE>
```

## Transfer

``` python
python unet_train.py <DATA_FILE>
```

