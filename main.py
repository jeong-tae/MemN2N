import numpy as np
import tensorflow as tf
import glob

from config import GeneralConfig
from util import parse_babi_task

from models import MemN2N_QA_Basic

data_dir = "data/tasks_1-20_v1-2/en"
task_id = 1

def run_task(data_dir, task_id):

    train_files = glob.glob('%s/qa%d_*_train.txt' % (data_dir, task_id))
    test_files = glob.glob('%s/qa%d_*_test.txt' % (data_dir, task_id))

    dictionary = {"nil" : 0}
    train_story, train_questions, train_qstory = parse_babi_task(train_files, dictionary, False)
    test_story, test_questions, test_qstory = parse_babi_task(test_files, dictionary, False)

    general_config = GeneralConfig(train_story, train_questions, dictionary)

    with tf.Session() as sess:
        model = MemN2N_QA_Basic(general_config, sess, (train_story, train_questions, train_qstory), (test_story, test_questions, test_qstory))

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
