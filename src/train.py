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
import os
os.chdir(r'D:\archive')
#%%
# data = pd.read_csv(r'D:\nlp_korea_bank\data\total_sample_labeling_fin44.csv', encoding='cp949')
# data = data[data['news/sentence'] == 0]
# df1 = data[data['소비자'] == 1][['content_new', '소비자']].iloc[:500] # 긍정
# df2 = data[data['소비자'] == 2][['content_new', '소비자']].iloc[:500] # 부정
# df = pd.concat([df1, df2], axis=0)
# df.head()
# #%%
# df.to_csv(r'D:\nlp_korea_bank\data\total_sample_labeling_small.csv', encoding='cp949')
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
'''CBOW context, target 생성'''
targets = [] 
contexts = []
all_contexts = []
labels = []
window_size = 4

for num in tqdm(range(len(sequences))):
    sent = sequences[num]
    l = len(sent) # 주어진 문장의 길이
    for index in range(l):
        s = index - window_size # window 시작 위치
        e = index + window_size + 1 # window 끝 위치
        context = []
        for i in range(s, e): 
            if 0 <= i < l and i != index: # window가 주어진 문장의 길이를 벗어나지 않고, 중심에 있는 단어(target)가 아닐 경우
                context.append(sent[i])
        contexts.append(context)
        targets.append(sent[index])
        all_contexts.append(sent)
        if sentiment[num] == 2: # 부정
            y = 0
        else: # 긍정
            y = 1
        labels.append(float(y))
        
print(len(contexts))
print(len(targets))
print(len(all_contexts))
print(len(labels))
#%%
contexts = tf.keras.preprocessing.sequence.pad_sequences(contexts, 
                                                        maxlen=window_size*2, 
                                                        dtype='int32', 
                                                        padding='post',
                                                        value=0)

all_contexts = tf.keras.preprocessing.sequence.pad_sequences(all_contexts, 
                                                        maxlen=all_context_window, 
                                                        dtype='int32', 
                                                        padding='post',
                                                        value=0)

targets = tf.one_hot(targets, depth=len(vocab))
#%%
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((contexts, targets, all_contexts, labels)).shuffle(len(contexts), reshuffle_each_iteration=True).batch(batch_size)
#%%
'''modeling'''
embedding_size = 32
def build_model(embedding_size):
    embedding_layer = K.layers.Embedding(len(vocab), embedding_size)
    output_layer = K.layers.Dense(len(vocab), activation='softmax')
    sentiment_layer = K.layers.Dense(1, activation='sigmoid')

    '''문맥 정보'''
    context_input = K.layers.Input((window_size*2, ))
    context_h = K.layers.GlobalAveragePooling1D()(embedding_layer(context_input)) # context embedding vector들을 평균

    all_context_input = K.layers.Input((all_context_window, ))
    all_context_h = K.layers.GlobalAveragePooling1D()(embedding_layer(all_context_input)) # 문장 embedding vector들을 평균

    h = context_h + all_context_h
    target_output = output_layer(h)

    '''감성 정보'''
    sentiment_output = sentiment_layer(all_context_h)

    model = K.models.Model([context_input, all_context_input], [target_output, sentiment_output])

    # model.summary()
    
    return model
#%%
optimizer = K.optimizers.Adam(0.001)
model = build_model(embedding_size)

@tf.function
def train_step(batch_context, batch_target, batch_all_contexts, batch_labels):
    with tf.GradientTape() as tape:
        context_pred, label_pred = model([batch_context, batch_all_contexts])
        context_loss = tf.reduce_mean(tf.reduce_sum(batch_target * tf.math.log(tf.clip_by_value(context_pred, 1e-10, 1.)), axis=1))
        sentiment_loss = batch_labels[:, tf.newaxis] * tf.math.log(tf.clip_by_value(label_pred, 1e-10, 1.))
        sentiment_loss += (1. - batch_labels[:, tf.newaxis]) * tf.math.log(tf.clip_by_value(1. - label_pred, 1e-10, 1.))
        sentiment_loss = tf.reduce_mean(sentiment_loss)
        loss = beta * context_loss + (1. - beta) * sentiment_loss
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    return loss, context_loss, sentiment_loss
#%%
iterator = iter(train_dataset)
beta = 0.99

iteration = 1000
step = 0
progress_bar = tqdm(range(iteration))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, iteration))

for _ in progress_bar:
    try:
        batch_context, batch_target, batch_all_contexts, batch_labels = next(iterator)
    except:
        iterator = iter(train_dataset)
        batch_context, batch_target, batch_all_contexts, batch_labels = next(iterator)
    
    loss, context_loss, sentiment_loss = train_step(batch_context, batch_target, batch_all_contexts, batch_labels)
    
    progress_bar.set_description('iteration {}/{} | loss {:.3f}, context {:.3f}, sentiment {:.3f}'.format(
        step, iteration, 
        loss.numpy(), context_loss.numpy(), sentiment_loss.numpy())) 
    
    step += 1
    
    if step == iteration: break
#%%
embedding_matrix = model.layers[2].weights[0]
embedding_matrix = embedding_matrix.numpy()
embedding_matrix.shape
'''save embedding matrix'''
np.save('./assets/embedding_matrix_{}'.format(beta), embedding_matrix)
#%%