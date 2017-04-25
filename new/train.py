#!/usr/bin/env python
# coding=utf-8
import time
import os
import tensorflow as tf
import tensorflow_fold.public.blocks as td
from utils import *
from model import DT_RNN

#Parameters
#======================================
#Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",100,"Dimensionality of word embedding(default:100)")




#Training parameters
tf.flags.DEFINE_integer("batch_size",10,"The size of a batch(default:10)")
tf.flags.DEFINE_integer("num_epochs",50,"Number of training epoch(default:50)")
tf.flags.DEFINE_string("save_path",'./tf_models/',"The path to save the models.")
tf.flags.DEFINE_string("data",'./data/hist_split',"The data to train.")
tf.flags.DEFINE_boolean("isTrain",True,"Is it training(default:True)")
FLAGS = tf.app.flags.FLAGS


#Load data
vocab, rel_list, ans_list, tree_dict = load_data(FLAGS.data)
node_dict = preprocess(tree_dict, array(ans_list), FLAGS.isTrain)
print "Load done..."


config = tf.ConfigProto( device_count = {'GPU': 0} )
sess = tf.InteractiveSession(config=config)
rnn = DT_RNN(FLAGS.embedding_dim, vocab, rel_list, FLAGS.isTrain)
train_op = tf.train.AdagradOptimizer(0.05).minimize(rnn.loss)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if not os.path.exists(FLAGS.save_path):
    os.mkdir(FLAGS.save_path)

min_loss = float(inf)
for _ in xrange(FLAGS.num_epochs):
  loss_sum = 0  
  for i in xrange(0,len(node_dict['train']),FLAGS.batch_size):
    batch = node_dict['train'][i:i+ FLAGS.batch_size]
    fdict = rnn.compiler.build_feed_dict(batch)
    begin = time.time()
    sess.run(train_op,feed_dict=fdict)
    end = time.time()
    loss_batch = sess.run(rnn.loss,feed_dict=fdict)
    loss_sum += loss_batch
    print "epoch :%d ,batch_id : %d ,time cost: %.4f,loss= %.4f" %(_, i/FLAGS.batch_size, end-begin,loss_batch)
  if loss_sum < min_loss:
    min_loss = loss_sum
  print "epoch loss : %.4f , min loss = %.4f." % (loss_sum, min_loss)
  saver.save(sess, FLAGS.save_path+"model.ckpt", global_step=_)
