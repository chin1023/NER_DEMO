# 获取词典

from Public.path import path_vocab

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 获取一般词典
def get_w2i(vocab_path = path_vocab):
    special_words = ['<PAD>', '<UNK>']
    with open(vocab_path, "r", encoding="utf-8") as f:
      char_vocabs = [line.strip() for line in f]
    char_vocabs = special_words + char_vocabs
    char_vocabs = sorted(set(char_vocabs), key = char_vocabs.index)
    i2w = {idx: char for idx, char in enumerate(char_vocabs)}
    w2i = {char: idx for idx, char in i2w.items()}
    return w2i


'''
# 获取 bert 词典
def get_w2i(vocab_path = path_vocab):
    w2i = {}
    with open(vocab_path, 'r') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i
'''


# 获取 tag to index 词典
def get_tag2index():
    return {'O':0,
            'ADMISSIONDATE-B': 1,     #住院日期-開頭
            'ADMISSIONDATE-I': 2,     #住院日期-以後
            'DISCHARGEDATE-B': 3,     #出院日期-開頭
            'DISCHARGEDATE-I': 4,     #出院日期-以後
            'SURGERYDATE-B': 5,       #手術日期-開頭
            'SURGERYDATE-I': 6,       #手術日期-以後
            'OUTPATIENTDATE-B': 7,    #門診日期-開頭
            'OUTPATIENTDATE-I': 8,    #門診日期-以後
            'CHEMOTHERAPYDATE-B': 9,  #化療日期-開頭
            'CHEMOTHERAPYDATE-I': 10, #化療日期-以後
            'RADIOTHERAPYDATE-B': 11, #放療日期-開頭
            'RADIOTHERAPYDATE-I': 12, #放療日期-以後
            'DISEASE-B': 13,      #疾病症狀-開頭
            'DISEASE-I': 14,      #疾病症狀-以後
            'TREATMENT-B': 15,    #處置方式-開頭
            'TREATMENT-I': 16,    #處置方式-以後
            'BODY-B': 17,         #器官部位-開頭
            'BODY-I': 18          #器官部位-以後
            }


'''
def get_tag2index():
    return {'O':0,
            'ADMISSIONDATE-B': 1,      #住院日期-開頭
            'ADMISSIONDATE-I': 2,      #住院日期-中間
            'ADMISSIONDATE-E': 3,      #住院日期-結尾
            'DISCHARGEDATE-B': 4,      #出院日期-開頭
            'DISCHARGEDATE-I': 5,      #出院日期-中間
            'DISCHARGEDATE-E': 6,      #出院日期-結尾
            'SURGERYDATE-B': 7,        #手術日期-開頭
            'SURGERYDATE-I': 8,        #手術日期-中間
            'SURGERYDATE-E': 9,        #手術日期-結尾
            'OUTPATIENTDATE-B': 10,    #門診日期-開頭
            'OUTPATIENTDATE-I': 11,    #門診日期-中間
            'OUTPATIENTDATE-E': 12,    #門診日期-結尾
            'CHEMOTHERAPYDATE-B': 13,  #化療日期-開頭
            'CHEMOTHERAPYDATE-I': 14,  #化療日期-中間
            'CHEMOTHERAPYDATE-E': 15,  #化療日期-結尾
            'RADIOTHERAPYDATE-B': 16,  #放療日期-開頭
            'RADIOTHERAPYDATE-I': 17,  #放療日期-中間
            'RADIOTHERAPYDATE-E': 18,  #放療日期-結尾
            'DISEASE-B': 19,      #疾病症狀-開頭
            'DISEASE-I': 20,      #疾病症狀-中間
            'DISEASE-E': 21,      #疾病症狀-結尾
            'TREATMENT-B': 22,    #處置方式-開頭
            'TREATMENT-I': 23,    #處置方式-中間
            'TREATMENT-E': 24,    #處置方式-結尾
            'BODY-B': 25,         #器官部位-開頭
            'BODY-I': 26,         #器官部位-中間
            'BODY-E': 27          #器官部位-結尾
            }
'''


if __name__ == '__main__':
    get_w2i()






















