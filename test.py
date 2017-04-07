#!/usr/bin/env python
# coding=utf-8
from numpy import *
from util.gen_util import *
from util.math_util import *
from util.dtree_util import *
import cPickle, time, argparse, random
import tensorflow as tf
import tensorflow_fold.public.blocks as td
def root(tree):
    return tree.get(0).kids[0][0]

def preprocess(tree_dict, ans_list):
    for key in tree_dict:
        for tree in tree_dict[key]:
            tree.get(root(tree)).ans = tree.ans_ind
            tree.get(root(tree)).dist = tree.dist
            tree.get(root(tree)).qid = tree.qid
            tree.get(root(tree)).neg = ans_list[ans_list != tree.ans_ind]
            for node in tree.get_nodes():
                new_kids = []
                for kid in node.kids:
                    new_kids.append((tree.get(kid[0]),kid[1]))
                node.kids = new_kids

def word(node):
    return node.ind

def get_kids(node):
    return node.kids

def get_ans(node):
    ans =[]
    ans.append(node.ans)
    return ans

def get_neg(node):
    neg = []
    neg.append(node.ans)
    random.shuffle(node.neg)
    neg_list = node.neg
    for i in xrange(100):
      neg.append(neg_list[i])
    return neg

vocab, rel_list, ans_list, tree_dict = cPickle.load(open('data/hist_split','r'))
ans_list = [vocab.index(ans) for ans in ans_list]

preprocess(tree_dict,array(ans_list))

node_dict = {}
for key in tree_dict:
    node_dict[key] = []
    for tree in tree_dict[key]:
        node_dict[key].append(root(tree))

with tf.device("/cpu:0"):
  w2v = tf.Variable(tf.random_normal([len(vocab),100,1]))


word2vec = (
           td.Optional(td.Scalar('int32')) >>
           td.Function(lambda x : tf.nn.embedding_lookup(w2v,x))
           )

Wv = td.FromTensor(tf.Variable(tf.random_normal([100,100])))


def rela(index):
    return Wr[rel_list.index(index)]

rel2mat = (td.InputTransform(rel_list.index) >>
           td.Optional(td.Scalar('int32') )  >>
           td.Function(td.Embedding(len(rel_list),100*100,name="rel_Matric")) >>
           td.Function(lambda x: tf.reshape(x,[-1,100,100])))

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


expr = td.ForwardDeclaration(td.PyObjectType(),td.TensorType([100,1]))
kids_deal = ( td.InputTransform(get_kids) >> td.Map(td.Record((expr(), rel2mat)))
              >> td.Map(Wr_mat) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((100,1)))) )
expr_def = td.AllOf(td.InputTransform(word) >> word2vec >> Wv_mat, kids_deal) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((100,1)))) >>td.Function(tf.tanh)
expr.resolve_to(expr_def)


compiler = td.Compiler.create(expr_def)
vec = compiler.output_tensors

sess = tf.InteractiveSession()
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state('tf_models')
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess,checkpoint.model_checkpoint_path)

train = sess.run(vec,feed_dict = compiler.build_feed_dict(node_dict['train']))
test = sess.run(vec,feed_dict = compiler.build_feed_dict(node_dict['test']))
print len(train)
