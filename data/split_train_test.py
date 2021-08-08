import pandas as pd
import random

def write_data(data,path):
    with open(path,'w') as f:
        for element in data:
            f.write(element + '\n')


data = pd.read_csv('triplet_data.csv')
anchor = data['anchor'].tolist()
positive = data['positive'].tolist()
negative = data['negative'].tolist()

label = []
sentence_1 = []
sentence_2 = []
for an, pos, neg in zip(anchor, positive, negative):
    sentence_1.append(an)
    sentence_2.append(pos)
    label.append('positive')

    sentence_1.append(an)
    sentence_2.append(neg)
    label.append('negative')
    
    sentence_1.append(pos)
    sentence_2.append(neg)
    label.append('neutral')

c = list(zip(sentence_1, sentence_2, label))
random.shuffle(c)
sentence_1, sentence_2, label = zip(*c)
ratio = 0.2

train_sentence_1 = sentence_1[:int(len(sentence_1)* (1- ratio))]
train_sentence_2 = sentence_2[:int(len(sentence_2)* (1- ratio))]
train_label = label[:int(len(label)* (1- ratio))]

valid_sentence_1 = sentence_1[int(len(sentence_1)* (1- ratio)):]
valid_sentence_2 = sentence_2[int(len(sentence_2)* (1- ratio)):]
valid_label = label[int(len(label)* (1- ratio)):]

print('# train sample', len(train_sentence_1))
print('# dev sample', len(valid_sentence_1))

write_data(train_sentence_1,'train/sentence_1.txt')
write_data(train_sentence_2,'train/sentence_2.txt')
write_data(train_label,'train/label.txt')


write_data(valid_sentence_1,'dev/sentence_1.txt')
write_data(valid_sentence_2,'dev/sentence_2.txt')
write_data(valid_label,'dev/label.txt')