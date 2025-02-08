#!/bin/bash
export NUMEXPR_MAX_THREADS=64
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export CUDA_VISIBLE_DEVICES=2


fairseq-hydra-train --config-dir config/pretrain/ --config-name hubert_base_librispeech.yaml task.data=/raid/home/rajivratn/hemant_rajivratn/librispeech/data/manifest/ task.label_dir=/raid/home/rajivratn/hemant_rajivratn/librispeech/stable_checkpoint_181_250000/fresh/iter3label/label/multiple_layers_960/ task.labels=[km500] model.label_rate=50 optimization.update_freq=[1] checkpoint.save_dir=/raid/home/rajivratn/hemant_rajivratn/weights/last/ +model.encoder_layers=12 optimization.max_update=400000

