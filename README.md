Due to the size limits, I could not upload the finetuned bert model and saved TSMIXER model. 

To run the project Download the missing folders from: 
https://drive.google.com/drive/folders/1Di09f_7f2wqDtsTu_p0yDpTDQf8H7lT1?usp=sharing/

Missing folders:
1. saved_model
2. finetuned_finbert_model

### Schema (The dataset folder can only contain the stock_price_data_raw & news_data_raw)
.
├── Pipfile
├── Pipfile.lock
├── app
│   ├── deployment_flow.py
│   ├── hyperparams.py
│   ├── main.py
│   ├── mlruns
│   │   ├── 0
│   │   │   └── meta.yaml
│   │   └── models
│   ├── model
│   │   ├── __init__.py
│   │   └── train.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   └── prepare_data.py
│   └── utils
│       ├── __init__.py
│       └── utils.py
├── dataset
│   ├── news_data_raw
│   │   └── aa.csv
│   ├── stock_price_data_raw
│   │   └── aa.csv
│   └── synthetic_news_sentiment_dataset.csv
├── end_to_end_TS.ipynb
├── finetuned_finbert_model
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── finetuned_finbert_model.zip
└── saved_model
    └── m_TSMixer.pkl
