import re
import numpy as np

f = open("concepts.csv", "r")
data_set = dict()
train_set = dict()
validation_set = dict()
test_set = dict()
max = 0
for line in f:
    image_name, list_concepts = line.split(':')
    list_concepts = re.sub(r"\n", "", list_concepts)
    concepts = list_concepts.split(';')
    for c in concepts:
        if c in data_set:
            old = []
            old = data_set[c]
            old.append(image_name)
            data_set[c] = old
        else:
            if max < 101:
                image=[]
                image.append(image_name)
                data_set[c] = image
                max += 1
f.close()

#Splitting the dataset into train, validation and test
for c in data_set:
    count = 0
    number_images = len(data_set[c])
    #TRAINING
    t =   int(round(number_images * 0.6))
    v = int(round((number_images * 0.2) + t))


    #All in training
    for image in data_set[c]:
        #TRAIN
        if count < t:
            if c in train_set:
                train_set[c] = train_set[c] + ";"+ image
            else :
                train_set[c] = image
                validation_set[c] = 'x'
                test_set[c] = 'x'
            count += 1
        #VAL
        elif count >= t and count < v:
            if c in validation_set:
                validation_set[c] = validation_set[c] + ";"+ image
            else :
                validation_set[c] = image
            count += 1
        #TEST
        else:
            if c in test_set:
                test_set[c] = test_set[c] + ";"+ image
            else :
                test_set[c] = image
            count += 1

######################## TRAINING
f = open("train_concepts.txt", "w")
for c in train_set:
    f.write(c+":")
    f.write(train_set[c]+"\n")
f.close()

######################## VALIDATION
f = open("validation_concepts.txt", "w")
for c in validation_set:
    f.write(c+":")
    f.write(validation_set[c]+"\n")
f.close()

######################## TEST
f = open("test_concepts.txt", "w")
for c in test_set:
    f.write(c+":")
    f.write(test_set[c]+"\n")
f.close()