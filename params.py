BASE="D:\\Acode\\labo\\aiprj"
#BASE="D:/Acode"
#BASE="/input/labo/aiprj"
#BASE="/autodl-fs/data/aiprj"
# bert模型参数
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-05
LABELS = ['non-rumor', 'false', 'true', 'unverified']
# 模型
MODEL_NAME_OR_PATH = BASE+"/bert_chinese"
HIDDEN_LAYER_SIZE = 768
