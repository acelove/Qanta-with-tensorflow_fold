#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import tensorflow_fold.public.blocks as td
import numpy as np
from utils import *
import cPickle
import os

class DT_RNN(object):
    def __init__( self, embedding_size, vocab, rel_list, isTrain, We_dir="./We"):

        #Model Definition
        r = sqrt(6) / sqrt(201)
        with tf.device("/cpu:0"):
          if os.path.exists(We_dir):
            We = cPickle.load(open(We_dir,'r'))
            W2V = random.rand(len(vocab), embedding_size,1)
            for i in xrange(len(vocab)):
              W2V[i] = We[:,i].reshape(embedding_size,1)
            w2v = tf.Variable(W2V.astype(float32))
          else:
            #w2v = tf.Variable(tf.random_normal([len(vocab), embedding_size, 1]))
            w2v = tf.Variable((random.rand(len(vocab),embedding_size,1)*2*r-r).astype(float32))

        #Wr = tf.Variable(tf.random_normal([len(rel_list), embedding_size, embedding_size])) 
        Wr = tf.Variable((random.rand(len(rel_list),embedding_size,embedding_size)*2*r-r).astype(float32))

        r = sqrt(6) / sqrt(51)
        #Wv = td.FromTensor(tf.Variable(tf.random_normal([embedding_size, embedding_size])))
        Wv = tf.Variable((random.rand(embedding_size,embedding_size)*2*r-r).astype(float32))
        #b = td.FromTensor(tf.Variable(tf.random_normal([embedding_size,1])))
        b = tf.Variable((random.rand(embedding_size,1)*2*r-r).astype(float32))

        word2vec = (
           td.Optional(td.Scalar('int32')) >>
           td.Function(lambda x : tf.nn.embedding_lookup(w2v,x))
           )

        rel2mat = (td.InputTransform(rel_list.index) >>
           td.Optional(td.Scalar('int32') )  >>
           td.Function(lambda x: tf.nn.embedding_lookup(Wr,x)))
           #td.Function(td.Embedding(len(rel_list), embedding_size*embedding_size, name="rel_Matric")) >>
           #td.Function(lambda x: tf.reshape(x,[-1, embedding_size, embedding_size])))


        Wr_mat = td.Composition()
        with Wr_mat.scope():
          vec = Wr_mat.input[0]
          WR = Wr_mat.input[1]
          out = td.Function(tf.matmul).reads(WR,vec)
          Wr_mat.output.reads(out)

        Wv_mat = td.Composition()
        with Wv_mat.scope():
          vec = Wv_mat.input
          Wv2 = td.FromTensor(Wv)
          out = td.Function(tf.matmul).reads(Wv2,vec)
          Wv_mat.output.reads(out)

        expression_label = (
                    td.Optional(td.Scalar('int32')) >>
                    td.Function(lambda x: tf.nn.embedding_lookup(w2v,x))
                   )

        loss = td.Composition()
        if isTrain:
            with loss.scope():
              vec0 = loss.input[0]
              vec1 = loss.input[1] 
              vec2 = loss.input[2]
              loss_pos = td.Function(lambda x,y :1 - tf.matmul(x,y,transpose_a=True)).reads(vec0,vec1) 
              vec0_b = td.Broadcast().reads(vec0)
              loss_pos_b = td.Broadcast().reads(loss_pos)
              loss_neg = (
                           td.Zip() >> td.Map(td.Function(lambda neg,ans,loss_pos  : tf.maximum(0.0, tf.matmul(neg,ans,transpose_a=True) + loss_pos )))
                            >> td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((1,1))))
                         ).reads(vec2,vec0_b,loss_pos_b)
              td.Metric('loss').reads(loss_neg)
              loss.output.reads(vec0)
        else:
            with loss.scope():
              vec = loss.input[0]
              flag = loss.input[1]
              ans = td.Function(lambda x,y :x*y).reads(vec,flag)
              td.Metric('loss').reads(ans)
              loss.output.reads(vec)



        expr = td.ForwardDeclaration(td.PyObjectType(),td.TensorType([embedding_size, 1]))
        kids_deal = ( 
                      td.InputTransform(get_kids) >> td.Map(td.Record((expr(), rel2mat)) ) 
                      >> td.Map(Wr_mat) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((embedding_size, 1))))
                    )
        pro = (
                 td.AllOf(td.InputTransform(word) >> word2vec >> Wv_mat , kids_deal) >>
                 td.Fold(td.Function(tf.add),td.FromTensor(b)) >>
                 td.Function(tf.tanh) >> 
                 td.Function(lambda x:x / tf.norm(x,axis=1,keep_dims = True))
              )
        expr_def = None
        if isTrain:
            expr_def = (
                        td.AllOf(pro,
                                 td.InputTransform(get_ans)>>td.Optional(td.Scalar('int32'))>>td.Function(lambda x: tf.nn.embedding_lookup(w2v,x)),
                                 td.InputTransform(get_neg) >> td.Map(expression_label))
                                 >>loss
                       )
        else:
            expr_def = td.AllOf(pro, td.InputTransform(isStop) >>td.Optional(td.Scalar('float32')) >> td.Function(lambda x:tf.reshape(x,[-1,1,1])) ) >> loss
        expr.resolve_to(expr_def)


        model = expr_def
        self.compiler = td.Compiler.create(model)
        self.vec_nodes = self.compiler.metric_tensors['loss']
        self.vec_h = self.compiler.output_tensors
        self.vec_word = None
        self.loss = None
        if isTrain:
            self.loss = tf.reduce_mean(self.vec_nodes) 
            print 
        else:
            vec_sum = tf.reduce_sum(self.vec_nodes,0)
            self.vec_word = vec_sum / tf.norm(vec_sum)
