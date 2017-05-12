#!/usr/bin/env python
# coding=utf-8
from util.gen_util import *
from util.math_util import *
from util.dtree_util import *
import numpy as np
import cPickle
from nltk.corpus import stopwords

def load_data(data_path):
    vocab, rel_list, ans_list, tree_dict = cPickle.load(open(data_path,'r'))
    ans_list = [vocab.index(ans) for ans in ans_list]
    return vocab, rel_list, ans_list, tree_dict

def root(tree):
    return tree.get(0).kids[0][0]

def preprocess(tree_dict, ans_list, isTrain):
    for key in tree_dict:
        for tree in tree_dict[key]:
            tree.get(root(tree)).dist = tree.dist
            tree.get(root(tree)).qid = tree.qid
            for node in tree.get_nodes():
                if isTrain:
                    node.ans = tree.ans_ind
                else:
                    node.ans = tree.ans
                node.neg = ans_list[ans_list != tree.ans_ind]
                new_kids = []
                for kid in node.kids:
                    new_kids.append((tree.get(kid[0]),kid[1]))
                node.kids = new_kids

    node_dict = {}
    for key in tree_dict:
        node_dict[key] = []
        for tree in tree_dict[key]:
            node_dict[key].append(root(tree))

    return node_dict

def word(node):
    return node.ind

def get_kids(node):
    return node.kids

def get_ans(node):
    return node.ans

def get_neg(node):
    neg = []
    neg.append(node.ans)
    neg_list = node.neg
    for i in xrange(100):
      neg.append(neg_list[i])
    return neg


def collapse_questions(train_trees, test_trees):
    train_q = {}
    for tree in train_trees:
        if tree.qid not in train_q:
            train_q[tree.qid] = {}
        train_q[tree.qid][tree.dist] = tree
    test_q = {}
    for tree in test_trees:
        if tree.qid not in test_q:
            test_q[tree.qid] = {}
        test_q[tree.qid][tree.dist] = tree
    return train_q, test_q

def shuffle_neg(trees):
    for tree in trees:
        for node in tree.get_nodes():
            random.shuffle(node.neg)

def isStop(node):
    stop = stopwords.words('english')
    if node.word in stop:
      return 0.0
    else:
      return 1.0
