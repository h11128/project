from math import log10
import string

items = []
with open('SMSSpamCollection', 'r', encoding='utf-8') as datafile: #, encoding='latin-1') as datafile:
    for line in datafile:
        row = line.rstrip().split('\t')
        # store first two fields as (label, message)
        items.append((row[0], row[1]))

# very simple tokenizer that first strips punctuation
punct_stripper = str.maketrans(dict.fromkeys(string.punctuation))
def tokenize(s):
    return s.translate(punct_stripper).split()

items = [(item[0], tokenize(item[1])) for item in items]

train_size = int(0.8 * len(items))
train_items, test_items = items[:train_size], items[train_size:]
y_train = []
x_train = []
for item in train_items:
    y_train.append(item[0])
    x_train.append(item[1])
y_test = []
x_test = []
for item in test_items:
    y_test.append(item[0])
    x_test.append(item[1])

def clean(list):
    final_list = []
    for i in list:
        final_list += i
    final_list = set(final_list)
    return final_list
words = clean(x_test + x_train)
"""# compute likelyhood for all single words p(x|y) WHERE x is whether a word appear or not"""
# dictionary for all the words. first one is num in ham, second is num in spam, third is p(x|ham)
# fourth is p(not x|ham), fifth is p(x|spam), sixth is p(not x|spam)
# count first and second
dicts = dict([(i,[0,0,0,0,0,0,0]) for i in words])
spam_num = 0
for i in range(len(y_train)):
    if y_train[i] =="ham":
        for j in (set(x_train[i])):
            dicts[j][0] +=  1
    else:
        spam_num += 1
        for j in (set(x_train[i])):
            dicts[j][1] +=  1
ham_num = len(y_train) - spam_num
p_ham = ham_num/len(y_train)
p_spam = 1- p_ham
# compute the last 4 terms
for i in dicts:
    dicts[i][2] = (dicts[i][0] / ham_num)
    dicts[i][3] = 1 - dicts[i][2]
    dicts[i][4] = (dicts[i][1] / spam_num)
    dicts[i][5] = 1 - dicts[i][4]
    if dicts[i][2] == 0:
        dicts[i][2] = 1/(ham_num +spam_num)
    if dicts[i][3] == 0:
        dicts[i][3] = 1/(ham_num +spam_num)
    if dicts[i][4] == 0:
        dicts[i][4] = 1/(ham_num +spam_num)
    if dicts[i][5] == 0:
        dicts[i][5] = 1/(ham_num +spam_num)
    dicts[i][6] = (dicts[i][0]+dicts[i][1])/(ham_num+spam_num)

"""predict from test set"""


p = []
print("            log likelihood of ham and spam")
for k in range(len(x_test)):
    p1 = log10(p_ham)
    p2 = log10(p_spam)
    for i in dicts:
        if i in set(x_test[k]):
            p1 += log10(dicts[i][2])
            p2 += log10(dicts[i][4])
        else:
            p1 += log10(dicts[i][3])
            p2 += log10(dicts[i][5])
    if p1>p2:
        p.append("ham")
    else: p.append("spam")
    if k <51:
        print("document",k,":",p1,p2,"  predict:",p[-1],"  fact:",y_test[k])
accuracy = 0
true = 0
for i in range(len(p)):
    if p[i] == y_test[i]:
        true += 1
accuracy = true/len(y_test)
print("accuracy: ",accuracy)
