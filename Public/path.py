import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址

# 词表目录
path_bert_vocab = os.path.join(current_dir, '../data/vocab/vocab.txt')  # bert预训练模型的词表
path_vocab = os.path.join(current_dir, '../data/vocab/char_vocabs_zh.txt')

# 实体命名识别文件目录
path_msra_dir = os.path.join(current_dir, '../data/MSRA/')
path_data300_dir = os.path.join(current_dir, '../data/data300/')
path_data600_dir = os.path.join(current_dir, '../data/data600/')
path_data300_bioes_dir = os.path.join(current_dir, '../data/data300_bioes/')

# save model
path_model = os.path.join(current_dir, '../Model/savemodel/')

# bert 预训练文件地址
path_bert_dir = os.path.join(current_dir, '../data/chinese_L-12_H-768_A-12/')

# 日志、记录类文件目录地址
path_log_dir = os.path.join(current_dir, "../log")

