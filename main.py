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
flags.DEFINE_float("max_grad", 40.0, "Gradient clipping")
flags.DEFINE_boolean("PE", True, "True, if you want to use position encoding, otherwise False")
flags.DEFINE_boolean("TE", True, "True, if you want to use temporal encoding, otherwise False")
flags.DEFINE_boolean("RN", True, "True, if you want to use random noise, otherwise False")
flags.DEFINE_string("weight_tying", "Adj", "Adj short for Adjacent, LW short for Layer-wise")

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

        model.build_model(task_id)
        model.run()

        model.writeResult(learning_result)


def main(_):

    all_tasks = True

    if all_tasks:
        for t in range(20):
            print(" [*] Task %d Learning Start" % (t+1))
            run_task(data_dir, t+1)    

if __name__ == '__main__':
    tf.app.run()
