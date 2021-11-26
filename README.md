# Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation
```
@misc{wang2019controllable,
      title={Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation}, 
      author={Ke Wang and Hang Hua and Xiaojun Wan},
      year={2019},
      eprint={1905.12926},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## librairies
```bash
import nltk
nltk.download('punkt')
```

## Data preprocessing
```bash
datapath=/content/imdb
references_files=""
data_columns=review,sentiment
save_to=/content

python preprocessed_data.py -f ${datapath}/data_train.csv,${datapath}/data_val.csv,${datapath}/data_test.csv -rf $references_files -dc $data_columns  -st $save_to
```

## Rename files
```bash
for data_type in train test val; do 
    mv ${save_to}/data_${data_type}_csv.csv ${save_to}/data_${data_type}.csv
done
```

## Train on train set (+ dev set if available) 
```bash
chmod +x main.sh
dump_path=/content
data_columns=review,sentiment
. main.sh $dump_path $data_columns
```

## Evaluate on test set
```bash
load_from_checkpoint=/content/pretrain/1
eval_only=True
. main.sh $dump_path $data_columns $load_from_checkpoint $eval_only
```

# References
- https://github.com/Nrgeup/controllable-text-attribute-transfer