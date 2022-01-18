#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
from konlpy.tag import Okt
from sklearn.decomposition import PCA
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
beta = 0.99
embedding_matrix = np.load('./assets/embedding_matrix_{}.npy'.format(beta))
#%%
pca = PCA(n_components=2)
pca.fit(embedding_matrix)
score = pca.transform(embedding_matrix)
#%%
count = 10
plt.figure(figsize=(6, 6))
plt.scatter(score[:count, 0], score[:count, 1], linewidths=1, color='blue', alpha=0.3)
plt.xlabel("PC1",size=15)
plt.ylabel("PC2",size=15)
plt.title("Word Embedding Space (beta={})".format(beta),size=15)
for word, i in vocab.items():
    if i == count: break
    plt.annotate(word, xy=(score[i, 0], score[i, 1]), fontsize=12)
plt.show()
#%%