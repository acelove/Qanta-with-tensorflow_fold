#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import tensorflow_fold.public.blocks as td
import numpy as np
from utils import *

class DT_RNN(object):
    def __init__(
            self, embedding_size, vocab, rel_list, isTrain 
            ):

        r = sqrt(6) / sqrt(201)
        #Model Definition
        with tf.device("/cpu:0"):
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
          out = td.Function(tf.matmul).reads(Wv,vec)
          Wv_mat.output.reads(out)

        expression_label = (
                    td.Optional(td.Scalar('int32')) >>
                    td.Function(lambda x: tf.nn.embedding_lookup(w2v,x))
                   )

        loss = td.Composition()
        if isTrain:
            with loss.scope():
              loss.output.reads(loss.input)
        else:
            with loss.scope():
              td.Metric('loss').reads(loss.input)
              loss.output.reads(loss.input)


        expr = td.ForwardDeclaration(td.PyObjectType(),td.TensorType([embedding_size, 1]))
        kids_deal = ( td.InputTransform(get_kids) >> td.Map(td.Record((expr(), rel2mat)))
                      >> td.Map(Wr_mat) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((embedding_size, 1))))
                    )
        #problem is the tanh fuunction
        pro = td.AllOf(td.InputTransform(word) >> word2vec >> Wv_mat, kids_deal) >>td.Fold(td.Function(tf.add),b) >>td.Function(tf.tanh) >> td.Function(lambda x:x / tf.norm(x, axis=1, keep_dims=True))
        expr_def = pro >> loss
        expr.resolve_to(expr_def)

        if isTrain:
          model = td.AllOf(expr_def, td.InputTransform(get_neg) >> td.Map(expression_label)>> td.NGrams(101) >> td.GetItem(0)) 
        else:
          model = expr_def
        self.compiler = td.Compiler.create(model)
        self.vec_h = self.compiler.output_tensors
        self.vec_word = None
        self.w2v = w2v
        self.Wr = Wr
        self.Wv = Wv
        self.b = b
        if not isTrain:
            self.vec_word = tf.reduce_mean(self.compiler.metric_tensors['loss'],0)
