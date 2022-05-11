import random
import numpy as np


def blocks_divider(array):
    x_ax = 0
    dum = list()
    for i in range(4):
        for y in range(0, len(array), 7):
            for x in range(x_ax, x_ax + 7):
                dum.append(list(array[x, y:y + 7]))

        x_ax += 7
    divided_photo = np.array(dum).reshape(16, 7, 7)
    return divided_photo


def pic_divider(x_train, x_test):
    train_vec = list()
    for i in x_train:
        im = blocks_divider(i)
        train_vec.append(im)
    test_vec = list()
    for i in x_test:
        im = blocks_divider(i)
        test_vec.append(im)
    return train_vec, test_vec


def center_calc(data):
    feature_vec = list()
    for image in data:
        vec = list()
        for block in image:
            x = 0
            y = 0
            sm = 0
            for y1 in range(7):
                for x1 in range(7):
                    x += x1 * block[y1][x1]
                    y += y1 * block[y1][x1]
                    sm += block[y1][x1]
            if sm == 0:
                x = 3
                y = 3
            else:
                x = x // sm
                y = y // sm
            vec.append(x)
            vec.append(y)
        feature_vec.append(vec)
    return feature_vec


def acc(y_test, predicted):
    sm = 0
    for i in range((len(predicted))):
        if y_test[i] == predicted[i]:
            sm += 1
    return (sm / len(y_test)) * 100


def target_processing(target):
    n_target = list()
    for i in target:
        dum = [0] * 10
        dum[i] = 1
        n_target.append(dum)
    return np.array(n_target)


def rand_lists(list_size, vector_size):
    vectors_list = list()
    for j in range(list_size):
        dum = list()
        for i in range(vector_size):
            dum.append(random.uniform(-2, 2))
        vectors_list.append(dum)
    return vectors_list
