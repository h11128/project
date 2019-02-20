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
#print('Probability of spam or ham /numbers of message:')
spam_num = 0
ham_num = 0
for item in train_items:
    if item[0] == 'spam':
        spam_num += 1
    else:
        ham_num += 1
p_spam = spam_num / len(train_items)
p_ham = ham_num / len(train_items)
print(p_spam,p_ham)

#count c(x1|ham) and c(x2|spam), store the value in two dicts
d_spam = {}
d_ham = {}
count = 0
for i in train_items:
    d_check = {}
    for word in train_items[count][1]:
        if(word in d_check):
            continue
        else:
            d_check[word] = 1
            if (train_items[count][0] == 'spam'):
                if(word in d_spam):
                    d_spam[word] += 1
                else:
                    d_spam[word] = 1
            if (train_items[count][0] == 'ham'):
                if(word in d_ham):
                    d_ham[word] += 1
                else:
                    d_ham[word] = 1
    count += 1

# count how many different words in train_set, named total_num
d_check = {}
total_num = 0
for item in train_items:
    for word in item[1]:
        if word in d_check :
            continue
        else:
            d_check[word] = 1
            total_num += 1
#print('total_num is :',total_num)


#Spam and ham number is spam_num, ham_num
#Calculate probability for test items, using d_ham and d_spam / number of ham or spam
right = 0
wrong = 0
show_num = 1
for item in test_items:

    isham = 1
    isspam = 1
    for word in item[1]:
        if word in d_ham:
            isham *= (d_ham[word])/(ham_num)
        if word not in d_ham:
            isham *= 1/(total_num)
        if word in d_spam:
            isspam *= (d_spam[word])/(spam_num)
        if word not in d_spam:
            isspam *= 1/(total_num)

    isham  *= p_ham
    isspam *= p_spam

    if(isham >= isspam):
        if(item[0] != 'ham'):
            wrong += 1
        else:
            right += 1
    if(isham < isspam):
        if(item[0] != 'spam'):
            wrong += 1
        else:
            right += 1
    if(show_num <= 50):
        print("p of ham is:", isham, ", p of spam is: ", isspam)
        if(isham >= isspam):
            print("Prediction: this document is a ham.")
        else:
            print("Prediction: this document is a spam.")
        print()
    show_num += 1

print('Prediction Accuracy：',(right/len(test_items)))
