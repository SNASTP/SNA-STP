# 训练配置
config_xor = {
    'architecture': [2, 2, 1],
    'activation': 'sigmoid',  # 或 'relu'
    'loss': 'binary_crossentropy',
    'optimizer': 'adam',
    'learning_rate': 0.1,
    'epochs': 1000,
    'batch_size': 4,
    'training_data': {
        'X': [[0,0], [0,1], [1,0], [1,1]],
        'y': [0, 1, 1, 0]
    }
}