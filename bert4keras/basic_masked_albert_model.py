import tensorflow as tf

from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])



base_path = 'D:\AI\Data\\albert_large_zh\\'
config_path = base_path + 'albert_config_large.json'
checkpoint_path = base_path + 'albert_model.ckpt'
dict_path = base_path + 'vocab.txt'

token_dict = load_vocab(dict_path) # 读取词典
tokenizer = SimpleTokenizer(token_dict) # 建立分词器
model = load_pretrained_model(config_path, checkpoint_path, albert=True) # 建立模型，加载权重


token_ids, segment_ids = tokenizer.encode(u'科学技术是第一生产力')

# mask掉“技术”
token_ids[3] = token_ids[4] = token_dict['[MASK]']

# 用mlm模型预测被mask掉的部分
probas = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
print(tokenizer.decode(probas[3:5].argmax(axis=1))) # 结果正是“技术”权