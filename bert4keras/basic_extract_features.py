import tensorflow as tf

from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

print(tf.__version__)

base_path = 'D:\AI\Data\chinese_L-12_H-768_A-12\\'
config_path = base_path + 'bert_config.json'
checkpoint_path = base_path + 'bert_model.ckpt'
dict_path = base_path + 'vocab.txt'

token_dict = load_vocab(dict_path) # 读取词典
tokenizer = SimpleTokenizer(token_dict) # 建立分词器
model = load_pretrained_model(config_path, checkpoint_path) # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))