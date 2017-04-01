from numpy import *
from util.gen_util import *
from util.math_util import *
from util.dtree_util import *
from rnn.adagrad import Adagrad
import rnn.propagation as prop
from classify.learn_classifiers import validate
import cPickle, time, argparse
from multiprocessing import Pool
import tensorflow as tf
import tensorflow_fold.public.blocks as td
import time
import random
def root(tree):
    return tree.get(0).kids[0][0]

def del1(tree_dict):
    for key in tree_dict:
        for tree in tree_dict[key]:
            for node in tree.get_nodes():
                new_kids = []
                for kid in node.kids:
                    new_kids.append((tree.get(kid[0]),kid[1]))
                node.kids = new_kids

def del2(tree_dict):
    for key in tree_dict:
        for tree in tree_dict[key]:
            root(tree).ans = tree.ans
            root(tree).dist = tree.dist
            root(tree).qid = tree.qid

def del3(tree_dict,ans_list):
    for key in tree_dict:
        for tree in tree_dict[key]:
            root(tree).neg = ans_list[ans_list != tree.ans]
 
def word(node):
    return node.word

def get_kids(node):
    return node.kids

def get_ans(node):
    ans =[]
    ans.append(node.ans)
    return ans

def get_neg(node):
    neg = []
    neg.append(vocab.index(node.ans))
    random.shuffle(node.neg)
    neg_list = node.neg
    for i in xrange(100):
      neg.append(vocab.index(neg_list[i]))
    return neg

vocab, rel_list, ans_list, tree_dict = cPickle.load(open('data/hist_split','r'))
del1(tree_dict)
del2(tree_dict)
del3(tree_dict,array(ans_list))
#val = tree_dict['dev']
#node_dict = del4(tree_dict)
node_dict = {}
for key in tree_dict:
    node_dict[key] = []
    for tree in tree_dict[key]:
        node_dict[key].append(root(tree))
with tf.device("/cpu:0"):
  w2v = tf.Variable(tf.random_normal([len(vocab),100,1]))
word2vec = (td.InputTransform(vocab.index) >>
           td.Optional(td.Scalar('int32')) >>
           td.Function(lambda x : tf.nn.embedding_lookup(w2v,x)) 
           #td.Function(td.Embedding(len(vocab),100,name="word_embed"))>>
           #td.Function(lambda x: tf.reshape(x,[-1,100,1]))
           )

Wv = td.FromTensor(tf.Variable(tf.random_normal([100,100])))
Wv2 = td.FromTensor(tf.Variable(tf.random_normal([100*100])))

Wr = {}
for rel in rel_list:
  Wr[rel] = td.FromTensor(tf.Variable(tf.random_normal([100,100])))

def rela(index):
    return Wr[rel_list.index(index)]

rel2mat = (td.InputTransform(rel_list.index) >>
           td.Optional(td.Scalar('int32') )  >>
           td.Function(td.Embedding(len(rel_list),100*100,name="rel_Matric")) >>
           td.Function(lambda x: tf.reshape(x,[-1,100,100])))

y = td.Composition()
with y.scope():
  vec = y.input[0]
  WR = y.input[1]
  #WR = td.Function(lambda x: tf.reshape(x,[-1,100,100])).reads(Wv2)
  #print WR
  out = td.Function(tf.matmul).reads(WR,vec)
  #out = td.Function(lambda x: tf.reshape(x,[-1,100,1])).reads(vec)
  #out = td.Identity().reads(vec)
  y.output.reads(out)

Wv_mat = td.Composition()
with Wv_mat.scope():
  vec = Wv_mat.input
  out = td.Function(tf.matmul).reads(Wv,vec)
  Wv_mat.output.reads(out)


expr = td.ForwardDeclaration(td.PyObjectType(),td.TensorType([100,1]))
kids_deal = ( td.InputTransform(get_kids) >> td.Map(td.Record((expr(), rel2mat)))
              >> td.Map(y) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((100,1)))) )
expr_def = td.AllOf(td.InputTransform(word) >> word2vec >> Wv_mat, kids_deal) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((100,1)))) >>td.Function(tf.tanh)
expr.resolve_to(expr_def)


#train = td.InputTransform(root) >> td.InputTransform(get_kids) >> td.Map(td.Record((td.InputTransform(word) >> word2vec, td.InputTransform(rel_list.index) >> td.InputTransform(rela)))) >> td.Map(y)
#sum = td.Fold(td.Function(tf.add), td.FromTensor(tf.zeros((100,))))
#ans = train>>sum  

expression_label = (
                    td.Optional(td.Scalar('int32')) >>
                    td.Function(lambda x: tf.nn.embedding_lookup(w2v,x)) 
                    #td.Function(lambda x: tf.reshape(x,[-1,100,1]))
                   )

#model = td.AllOf(expr_def, td.InputTransform(get_neg) >> td.Map(expression_label)>> td.NGrams(3) >> td.GetItem(0) >>td.Concat()>>td.Function(lambda x:tf.reshape(x,[-1,3,100,1]))) 
model = td.AllOf(expr_def, td.InputTransform(get_neg) >> td.Map(expression_label)>> td.NGrams(101) >> td.GetItem(0)) 
compiler = td.Compiler.create(model)
vec = compiler.output_tensors


#cos = tf.matmul(vec[0],vecs[:,0],transpose_a=True) / (tf.norm(vec[0],axis=1,keep_dims=True) * tf.norm(vecs[:,0],axis=1,keep_dims=True))
cos = tf.matmul(vec[0],vec[1],transpose_a=True) / (tf.norm(vec[0],axis=1,keep_dims=True) * tf.norm(vec[1],axis=1,keep_dims=True))

loss_pos = 1 - cos
loss = loss_pos

for i in xrange(100):
    loss = loss + tf.maximum(0.0, tf.matmul(vec[0],vec[i+2],transpose_a=True) / (tf.norm(vec[0],axis=1,keep_dims=True) * tf.norm(vec[i+2],axis=1,keep_dims=True)) + loss_pos)

loss = tf.reduce_sum(loss)

train_op = tf.train.AdagradOptimizer(0.05).minimize(loss)
saver = tf.train.Saver()
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
    
#batch = []
#for i in xrange(272):
#    batch.append(root(val[i]))
#fdict = compiler.build_feed_dict(batch)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
#config.log_device_placement=True
allow_soft_placement=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

#begin = time.time()
min_loss = 1000000000.0
for _ in xrange(80):
  #sess.run(train_op,feed_dict=fdict)
  #print sess.run(vec,feed_dict = fdict)
  loss_sum = 0  
  for i in xrange(0,len(tree_dict['train']),272):
    #batch = []
    #for j in xrange(272):
      #batch.append(root(tree_dict['train'][i+j]))
    batch = node_dict['train'][i:i+272]
    #print len(batch)
    #print batch[0]
    fdict = compiler.build_feed_dict(batch)
    begin = time.time()
    sess.run(train_op,feed_dict=fdict)
    end = time.time()
    loss_batch = sess.run(loss,feed_dict=fdict)
    loss_sum += loss_batch
    #print linalg.norm(sess.run(vec,feed_dict = fdict))
    print "epoch :%d ,batch_id : %d ,time cost: %.4f,loss= %.4f" %(_, i/272, end-begin,loss_batch)
  if loss_sum < min_loss:
    min_loss = loss_sum
  print "epoch loss : %.4f , min loss = %.4f." % (loss_sum, min_loss)
  saver.save(sess,"./tf_models/model.ckpt",global_step=_)
