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
            #tree.get(root(tree)).ans = tree.ans_ind
            tree.get(root(tree)).dist = tree.dist
            tree.get(root(tree)).qid = tree.qid
            #tree.get(root(tree)).neg = ans_list[ans_list != tree.ans_ind]
            for node in tree.get_nodes():
                node.ans = tree.ans_ind
                node.neg = ans_list[ans_list != tree.ans_ind]
                new_kids = []
                for kid in node.kids:
                    new_kids.append((tree.get(kid[0]),kid[1]))
                node.kids = new_kids

def word(node):
    return node.ind

def get_kids(node):
    return node.kids

def get_ans(node):
    return node.ans

def get_neg(node):
    neg = []
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
b = td.FromTensor(tf.Variable(tf.random_normal([100,1])))

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
  #zero = td.FromTensor(tf.zeros((100,1)))
  vec0 = loss.input[0]
  vec1 = loss.input[1] 
  vec2 = loss.input[2]
  #vec = td.Function(lambda x:tf.transpose(x,perm=[0,2,1])).reads(vec0)
  #loss = td.Map(td.Function(lambda x:tf.matmul(x,x,transpose_a=True))).reads(vec2)
  #loss2 = td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((100,1)))).reads(loss)
  loss_pos = td.Function(lambda x,y :1 - tf.matmul(x,y,transpose_a=True)).reads(vec0,vec1) 
  vec0_b = td.Broadcast().reads(vec0)
  loss_pos_b = td.Broadcast().reads(loss_pos)
  loss_neg = (
               td.Zip() >> td.Map(td.Function(lambda neg,ans,loss_pos : tf.maximum(0.0, tf.matmul(neg,ans,transpose_a=True) + loss_pos)))
               >> td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((1,1))))
             ).reads(vec2,vec0_b,loss_pos_b)
  td.Metric('loss').reads(loss_neg)
  loss.output.reads(vec0)



expr = td.ForwardDeclaration(td.PyObjectType(),td.TensorType([100,1]))
kids_deal = ( td.InputTransform(get_kids) >> td.Map(td.Record((expr(), rel2mat)))
              >> td.Map(Wr_mat) >>td.Fold(td.Function(tf.add),td.FromTensor(tf.zeros((100,1)))) )
pro = td.AllOf(td.InputTransform(word) >> word2vec >> Wv_mat, kids_deal) >>td.Fold(td.Function(tf.add),b) >>td.Function(tf.tanh) >> td.Function(lambda x:x / tf.norm(x))
expr_def = td.AllOf(pro,td.InputTransform(get_ans)>>td.Optional(td.Scalar('int32')) >>td.Function(lambda x: tf.nn.embedding_lookup(w2v,x)), td.InputTransform(get_neg) >> td.Map(expression_label)) >>loss
expr.resolve_to(expr_def)

 
model = expr_def
compiler = td.Compiler.create(model)
vec = compiler.metric_tensors['loss']


loss = tf.reduce_sum(vec) 

train_op = tf.train.AdagradOptimizer(0.05).minimize(loss)
saver = tf.train.Saver()


#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
#config.gpu_options.allow_growth = True
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )


#allow_soft_placement=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())


min_loss = 1000000000.0
for _ in xrange(50):
  loss_sum = 0  
  for i in xrange(0,len(tree_dict['train']),10):
    batch = node_dict['train'][i:i+ 10]
    fdict = compiler.build_feed_dict(batch)
    begin = time.time()
    sess.run(train_op,feed_dict=fdict)
    end = time.time()
    loss_batch = sess.run(loss,feed_dict=fdict)
    loss_sum += loss_batch
    print "epoch :%d ,batch_id : %d ,time cost: %.4f,loss= %.4f" %(_, i/10, end-begin,loss_batch)
  if loss_sum < min_loss:
    min_loss = loss_sum
  print "epoch loss : %.4f , min loss = %.4f." % (loss_sum, min_loss)
  saver.save(sess,"./tf_models/model.ckpt",global_step=_)
