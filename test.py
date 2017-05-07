
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

tf.flags.DEFINE_integer("embedding_dim",100,"Dimensionality of word embedding(default:100)")


tf.flags.DEFINE_integer("batch_size",11,"The size of a batch(default:11)")
tf.flags.DEFINE_integer("num_epochs",50,"Number of training epoch(default:50)")
tf.flags.DEFINE_string("save_path",'./tf_models/',"The path to save the models.")
tf.flags.DEFINE_string("data",'./data/hist_split',"The data to train.")
tf.flags.DEFINE_boolean("isTrain",True,"Is it training(default:True)")
tf.flags.DEFINE_string("model_path",'./tf_models/',"The path to load the model.")
FLAGS = tf.app.flags.FLAGS

config = tf.ConfigProto( device_count = {'GPU': 0} )
sess = tf.InteractiveSession(config=config)
FLAGS.isTrain=False
vocab, rel_list, ans_list, tree_dict = load_data(FLAGS.data)
node_dict = preprocess(tree_dict, array(ans_list), FLAGS.isTrain)

rnn = DT_RNN(FLAGS.embedding_dim, vocab, rel_list, FLAGS.isTrain)
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(FLAGS.model_path)
saver.restore(sess, checkpoint.model_checkpoint_path)
node = None
for tree in tree_dict['train']:
  if tree.qid==1490 and tree.dist==0:
    node =tree


for ns in node.get_nodes():
  for kid in ns.kids:
    if kid[0].word == "north":
      node = ns


batch = []
batch.append(node)
fdict = rnn.compiler.build_feed_dict(batch)
sess.run(rnn.vec_nodes,feed_dict=fdict)
