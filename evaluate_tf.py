#!/usr/bin/env python
# coding=utf-8
from numpy import *
import nltk.classify.util
from util.gen_util import *
from util.math_util import *
from util.dtree_util import *
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import cPickle
import tensorflow as tf
import tensorflow_fold.public.blocks as td


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

def root(tree):
    return tree.get(0).kids[0][0]

def preprocess(tree_dict, ans_list):
    for key in tree_dict:
        for tree in tree_dict[key]:
            #tree.get(root(tree)).ans = tree.ans_ind
            tree.get(root(tree)).dist = tree.dist
            tree.get(root(tree)).qid = tree.qid
            #tree.get(root(tree)).neg = ans_list[ans_list != tree.ans_ind]
            for node in tree.get_nodes():
                node.ans = tree.ans
                node.neg = ans_list[ans_list != tree.ans_ind]
                new_kids = []
                for kid in node.kids:
                    new_kids.append((tree.get(kid[0]),kid[1]))
                node.kids = new_kids

def word(node):
    return node.ind

def get_kids(node):
    return node.kids

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

loss = td.Composition()


expression_label = (
                    td.Optional(td.Scalar('int32')) >>
                    td.Function(lambda x: tf.nn.embedding_lookup(w2v,x))
                   )


with loss.scope():
  td.Metric('loss').reads(loss.input)
  loss.output.reads(loss.input)



expr = td.ForwardDeclaration(td.PyObjectType(),td.TensorType([100,1]))
kids_deal = ( td.InputTransform(get_kids) >> td.Map(td.Record((expr(), rel2mat)))
              >> td.Map(Wr_mat) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((100,1)))) )
pro = td.AllOf(td.InputTransform(word) >> word2vec >> Wv_mat, kids_deal) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((100,1)))) >>td.Function(tf.tanh) >> td.Function(lambda x:x / tf.norm(x))
expr_def = pro >>loss
expr.resolve_to(expr_def)

 
model = expr_def
compiler = td.Compiler.create(model)
vec_nodes = compiler.metric_tensors['loss']
vec_word = tf.reduce_mean(vec_nodes,0)
vec_h = compiler.output_tensors

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )


#allow_soft_placement=True
sess = tf.InteractiveSession(config=config)
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state("tf_models")
if checkpoint and checkpoint.model_checkpoint_path:
     saver.restore(sess, checkpoint.model_checkpoint_path)
     print "Successfully loaded:", checkpoint.model_checkpoint_path





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
      fdict = compiler.build_feed_dict(tmp[0:1])
      h = sess.run(vec_h,feed_dict=fdict)
      words = sess.run(vec_word,feed_dict=fdict)
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
