#!/bin/bash

# . main.sh $dump_path data_columns
dump_path=${1-save}
data_path=/content
train_data_file=${data_path}/data_train.csv
val_data_file=${data_path}/data_val.csv
test_data_file=${data_path}/data_test.csv
references_files=,
word_to_id_file=${data_path}/word_to_id.txt
data_columns=$2
batch_size=128
if_load_from_checkpoint=${3-False}
checkpoint_name=${4-None}
eval_only=${5-False}

if [ $if_load_from_checkpoint = "True" ]; then
    if [ $checkpoint_name = "None" ]; then
        echo "Error, give a valid checkpoint name"
        exit
    fi
fi

python3 main.py \
		--id_pad 0 \
		--id_unk 1 \
		--id_bos 2 \
		--id_eos 3 \
		--dump_path $dump_path \
		--data_path $data_path \
		--train_data_file $train_data_file \
		--val_data_file $val_data_file \
		--test_data_file $test_data_file \
		--references_files $references_files \
		--word_to_id_file $word_to_id_file \
		--data_columns $data_columns \
		--word_dict_max_num 5 \
		--batch_size $batch_size \
		--max_sequence_length 60 \
		--num_layers_AE 2 \
		--transformer_model_size 256 \
		--transformer_ff_size 1024 \
		--latent_size 256 \
		--word_dropout 1.0 \
		--embedding_dropout 0.5 \
		--learning_rate 0.001 \
		--label_size 1 \
		--max_epochs 10 \
		--log_interval 100 \
		--eval_only $eval_only \
		--if_load_from_checkpoint $if_load_from_checkpoint \
		--checkpoint_name $checkpoint_name