import numpy as np
import tensorflow as tf
import glob

from util import parse_babi_task

from models import MemN2N_QA_Basic

flags = tf.app.flags
flags.DEFINE_integer("max_epoch", 100, "Epoch to train")
flags.DEFINE_integer("batch_size", 32, "Number of instance for 1 iteration")
flags.DEFINE_integer("max_mem_size", 50, "The size of memory")
flags.DEFINE_integer("edim", 20, "Embedding dimension")
flags.DEFINE_integer("nhops", 3, "Number of memory layers to hop")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate to train")
flags.DEFINE_float("lr_anneal", 25, "Every epoch to anneal learning rate")
flags.DEFINE_float("init_std", 0.1, "Standard deviation of random variable with initialization")
flags.DEFINE_float("max_grad_norm", 40.0, "Gradient clipping")
flags.DEFINE_boolean("is_test", False, "False if you want to train")
flags.DEFINE_boolean("PE", True, "True, if you want to use position encoding, otherwise False")
flags.DEFINE_boolean("TE", True, "True, if you want to use temporal encoding, otherwise False")
flags.DEFINE_boolean("LS", True, "True, if you want to use linear start(training recommence by re-inserting a softmax layer)")
flags.DEFINE_boolean("RN", False, "True, if you want to use random noise, otherwise False")
flags.DEFINE_string("weight_tying", "Adj", "Adj short for Adjacent, LW short for Layer-wise")
flags.DEFINE_string("checkpoint", "checkpoints/model.ckpt", "Path for the pre-trained model")

FLAGS = flags.FLAGS

data_dir = "data/tasks_1-20_v1-2/en"
task_id = 1

def run_task(data_dir, task_id):

    train_files = glob.glob('%s/qa%d_*_train.txt' % (data_dir, task_id))
    test_files = glob.glob('%s/qa%d_*_test.txt' % (data_dir, task_id))

    dictionary = {"nil" : 0}
    train_story, train_questions, train_qstory = parse_babi_task(train_files, dictionary, False)
    test_story, test_questions, test_qstory = parse_babi_task(test_files, dictionary, False)
    FLAGS.dictionary = dictionary

    with tf.Session() as sess:
        model = MemN2N_QA_Basic(FLAGS, sess, (train_story, train_questions, train_qstory), (test_story, test_questions, test_qstory))

        # only qa_model in this repository yet
        model.qa_model(task_id)
        model.optimization()

        model.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(model.sess, ckpt.model_checkpoint_path)
        else:
            print(" [!] Not found checkpoint")

        sess.run(tf.global_variables_initializer())

        if FLAGS.is_test:
            model.test()
        else:
            model.train()

    tf.reset_default_graph()
    sess.close()

def main(_):

    all_tasks = True

    if all_tasks:
#for t in range(20):
#print(" [*] Task %d Learning Start" % (t+1))
            run_task(data_dir, 2)    

if __name__ == '__main__':
    tf.app.run()
