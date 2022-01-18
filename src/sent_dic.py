#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
from konlpy.tag import Okt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정
import os
os.chdir(r'D:\archive')
#%%
df = pd.read_csv(r'D:\nlp_korea_bank\data\total_sample_labeling_small.csv', encoding='cp949')
#%%
corpus = df['content_new'].to_list()
sentiment = df['소비자'].to_list()
#%%
def clean_korean(sent):
    if type(sent) == str:
        h = re.compile('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]+')
        result = h.sub(' ', sent)
        result=result.replace('\n','').strip()
    else:
        result = ''
    return result

corpus_clean = [clean_korean(i) for i in corpus]
#%%
okt = Okt()
use_tag = ['Noun', 'Verb', 'Adjective']
preprocess_corpus = []
for x, y in zip(corpus_clean, sentiment):
    sent = okt.pos(x)
    if y == 2: y = 0
    sent = [s[0] + '_' + str(y) for s in sent if s[1] in use_tag and len(s[0]) > 1]
    preprocess_corpus.append(sent)
#%%
from tensorflow.keras import preprocessing
tokenizer=preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(preprocess_corpus)
sequences=tokenizer.texts_to_sequences(preprocess_corpus)
vocab=tokenizer.word_index
vocab['pad'] = 0

vocab = {x:y for x,y in sorted(vocab.items(), key=lambda x: x[1])}
#%%
plt.hist([len(x) for x in sequences])
plt.title('sequence length histogram')
plt.show()

# from scipy.stats import mode
all_context_window = np.max([len(x) for x in sequences])
#%%
vocab_reverse = {i:x for x,i in vocab.items()}
vocab_size = len(vocab)
#%%
'''load embedding matrix'''
beta = 0.3
embedding_matrix = np.load('./assets/embedding_matrix_{}.npy'.format(beta))
#%%
'''find similar words with cosine similarity'''
F2norm = np.linalg.norm(embedding_matrix, axis=1)
topk = 10
target_word = '소비_1'
assert target_word in vocab.keys()
u = embedding_matrix[vocab.get(target_word), :]
sim = np.matmul(embedding_matrix, u) / (F2norm * np.linalg.norm(u))
topkidx = np.argsort(sim)[::-1][1:topk+1]
print('topk similar words of {}:'.format(target_word), [(vocab_reverse.get(i), sim[i]) for i in topkidx])
#%%
target_word = '소비_0'
assert target_word in vocab.keys()
u = embedding_matrix[vocab.get(target_word), :]
sim = np.matmul(embedding_matrix, u) / (F2norm * np.linalg.norm(u))
topkidx = np.argsort(sim)[::-1][1:topk+1]
print('topk similar words of {}:'.format(target_word), [(vocab_reverse.get(i), sim[i]) for i in topkidx])
#%%