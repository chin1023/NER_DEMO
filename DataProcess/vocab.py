# 获取词典

from Public.path import path_vocab

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 获取 word to index 词典
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


if __name__ == '__main__':
    get_w2i()






















