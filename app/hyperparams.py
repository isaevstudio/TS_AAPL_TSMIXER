# Hyperparameters
configs = {
    "data": {
        "sequence_length": 50,
        "target_column": "close"
    },
    "training": {
        "epochs": 20,
        "batch_size": 32
    },
    "model": {
        "loss": "mse",
        "optimizer": "adam",
        "save_dir": "saved_models",
        "layers": [
            {
                "type": "lstm",
                "neurons": 100,
                "return_seq": True
            },
            {"type": "dropout", "rate": 0.2},
            {
                "type": "lstm",
                "neurons": 100,
                "return_seq": False
            },
            {"type": "dropout", "rate": 0.2},
            {"type": "dense", "neurons": 1, "activation": "linear"}
        ]
    }
}