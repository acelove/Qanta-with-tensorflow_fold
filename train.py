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

#Parameters
#======================================
#Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",100,"Dimensionality of word embedding(default:100)")


#Program parameters
tf.flags.DEFINE_integer("batch_size",10,"The size of a batch(default:10)")
tf.flags.DEFINE_integer("num_epochs",50,"Number of training epoch(default:50)")
tf.flags.DEFINE_string("save_path",'./tf_models/',"The path to save the models.")
tf.flags.DEFINE_string("data",'./data/hist_split',"The data to train.")
tf.flags.DEFINE_boolean("isTrain",True,"Is it training(default:True)")
tf.flags.DEFINE_string("model_path",'./tf_models/',"The path to load the model.")
FLAGS = tf.app.flags.FLAGS

#Build a session
config = tf.ConfigProto( device_count = {'GPU': 0} )
sess = tf.InteractiveSession(config=config)

#train
def train(rnn, saver, node_dict):
    train_op = tf.train.AdagradOptimizer(0.05).minimize(rnn.loss)
    sess.run(tf.global_variables_initializer())
    if not os.path.exists(FLAGS.save_path):
        os.mkdir(FLAGS.save_path)

    min_loss = float(inf)
    for _ in xrange(FLAGS.num_epochs):
      loss_sum = 0  
      for i in xrange(0,len(node_dict['train']),FLAGS.batch_size):
        batch = node_dict['train'][i:i+FLAGS.batch_size]
        fdict = rnn.compiler.build_feed_dict(batch)
        begin = time.time()
        sess.run(train_op,feed_dict=fdict)
        end = time.time()
        loss_batch = sess.run(rnn.loss,feed_dict=fdict)
        loss_sum += loss_batch
        print "epoch :%d ,batch_id : %d ,time cost: %.4f,loss= %.4f" %(_, i/FLAGS.batch_size, end-begin,loss_batch)
      if loss_sum < min_loss:
        min_loss = loss_sum
        saver.save(sess, FLAGS.save_path+"model.ckpt",global_step=_)
        print "saving model"
      print "epoch loss : %.4f , min loss = %.4f." % (loss_sum, min_loss)

def evaluate(rnn, saver, node_dict):
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print "Successfully load model:", checkpoint.model_checkpoint_path

    train_nodes = node_dict['train']
    test_nodes = node_dict['test']
    train_q, test_q = collapse_questions(train_nodes, test_nodes)

    train_feats = []
    test_feats = []

    for tt, split in enumerate([train_q, test_q]):
      for qid in split:
        q = split[qid]
        curr_ave = zeros ( (100, 1))
        curr_words = zeros ( (100, 1))
        for i in range(0, len(q)):
          tree = q[i]
          tmp = []
          tmp.append(tree)
          fdict = rnn.compiler.build_feed_dict(tmp[0:1])
          h = sess.run(rnn.vec_h,feed_dict=fdict)
          words = sess.run(rnn.vec_word,feed_dict=fdict)
          curr_ave += array(h).reshape(100,1)
          curr_ave = curr_ave / linalg.norm(curr_ave)
          curr_words += array(words).reshape(100,1)
          curr_words = curr_words / linalg.norm(curr_words)
        featvec = concatenate([curr_ave.flatten(), curr_words.flatten()])
        curr_feats = {}
        for dim, val in ndenumerate(featvec):
            curr_feats['__' + str(dim)] = val
        if tt == 0:
            train_feats.append( (curr_feats, tree.ans.lower()) )
        else:
            test_feats.append( (curr_feats, tree.ans.lower()) )
    print 'total training instances:', len(train_feats)
    print 'total testing instances:', len(test_feats)
   
    classifier = SklearnClassifier(LinearSVC(C=10))
    classifier.train(train_feats)

    print 'accuracy train:', nltk.classify.util.accuracy(classifier, train_feats)
    print 'accuracy test:', nltk.classify.util.accuracy(classifier, test_feats)


def main(_):
  #Load data
  vocab, rel_list, ans_list, tree_dict = load_data(FLAGS.data)
  node_dict = preprocess(tree_dict, array(ans_list), FLAGS.isTrain)
  print "Load data finish..."
  rnn = DT_RNN(FLAGS.embedding_dim, vocab, rel_list, FLAGS.isTrain)
  saver = tf.train.Saver()
  if FLAGS.isTrain:
      train(rnn, saver, node_dict)
  else:
      evaluate(rnn, saver, node_dict)

if __name__ == "__main__":
    tf.app.run()
