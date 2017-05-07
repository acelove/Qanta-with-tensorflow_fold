#!/usr/bin/env python
# coding=utf-8
import time
import os
import tensorflow as tf
import tensorflow_fold.public.blocks as td
from utils import *
from model import DT_RNN
from numpy import *
import nltk.classify.util
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
import sys

#Parameters
#======================================
#Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",100,"Dimensionality of word embedding(default:100)")


#Program parameters
tf.flags.DEFINE_integer("batch_size",11,"The size of a batch(default:11)")
tf.flags.DEFINE_integer("num_epochs",50,"Number of training epoch(default:50)")
tf.flags.DEFINE_string("save_path",'./tf_models/',"The path to save the models.")
tf.flags.DEFINE_string("data",'./data/hist_split',"The data to train.")
tf.flags.DEFINE_boolean("isTrain",True,"Is it training(default:True)")
tf.flags.DEFINE_string("model_path",'./tf_models/',"The path to load the model.")
FLAGS = tf.app.flags.FLAGS

#Build a session
config = tf.ConfigProto( device_count = {'GPU': 0} )
sess = tf.InteractiveSession(config=config)

vocab, rel_list, ans_list, tree_dict = load_data(FLAGS.data)
node_dict = preprocess(tree_dict, array(ans_list), FLAGS.isTrain)
print "Load data finish..."
rnn = DT_RNN(FLAGS.embedding_dim, vocab, rel_list, FLAGS.isTrain)
saver = tf.train.Saver()

checkpoint = tf.train.get_checkpoint_state(FLAGS.model_path)
saver.restore(sess, checkpoint.model_checkpoint_path)

import cPickle
w2v = rnn.w2v.eval()
Wr = rnn.Wr.eval()
Wv = rnn.Wv.eval()
b = rnn.b.eval()

W2V = zeros([100,len(w2v)])
for i in xrange(len(w2v)):
    W2V[:,i] = w2v[i].reshape(100)

cPickle.dump((W2V,Wr,Wv,b,rel_list),open('datas','w'))
