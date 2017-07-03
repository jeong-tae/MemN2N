import random
import numpy as np
import tensorflow as tf
from tqdm import trange
import os
import sys
import time
from ops import *

random.seed(1003)
np.random.seed(1003)

class MemN2N_QA_Basic(object): # m_i = A * x_i, version of Bow
    def __init__(self, config, sess, train, test):
        
        self.sess = sess
        self.train_story, self.train_questions, self.train_qstory = train
        self.test_story, self.test_questions, self.test_qstory = test

        self.lr = tf.Variable(config.learning_rate, trainable = False, name = 'lr')
        self.lr_anneal = config.lr_anneal
        self.batch_size = config.batch_size
        self.nhops = config.nhops
        self.init_std = config.init_std
        self.max_epoch = config.max_epoch
        self.dictionary = config.dictionary
        self.voca_size = len(self.dictionary)
        self.max_mem_size = min(config.max_mem_size, self.train_story.shape[1])
        self.max_words = len(self.train_story) # we can define separated buckets for this
        self.edim = config.edim
        self.max_grad_norm = config.max_grad_norm
        self.PE = config.PE
        self.TE = config.TE
        self.LS = config.LS
        self.RN = config.RN
        self.weight_tying = config.weight_tying

		# self.display_interval = config.display_interval
		# self.test_interval = config.test_interval

        self.train_len = self.train_questions.shape[1]
        self.test_len = self.test_questions.shape[1]

        self.input_querys = tf.placeholder(tf.int32, [None, 1, self.max_words])
        self.input_episodes = tf.placeholder(tf.int32, [None, self.max_mem_size, self.max_words])
        self.labels = tf.placeholder(tf.float32, [None, self.voca_size])

    def get_variable(self, name, shape, stddev = None, scope = None, reuse = False):
        
        if scope:
            pass
        else:
            if stddev != None and reuse == False:
                w = tf.get_variable(name, shape, initializer = tf.truncated_normal_initializer(stddev = stddev), trainable = True)
                return w
            elif reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
                    w = tf.get_variable(name, trainable = True)
                return w

    def qa_model(self, task_id):
        # M = Memory
        self.task_id = task_id
        print(" [*] Model building...")
        emb_A = self.get_variable("A1", shape = [self.voca_size, self.edim], stddev = self.init_std)
        emb_B = self.get_variable("B1", shape = [self.voca_size, self.edim], stddev = self.init_std)
        emb_C = self.get_variable("C1", shape = [self.voca_size, self.edim], stddev = self.init_std)

        # m_i = sum A_ij * x_ij
        mem_M = tf.nn.embedding_lookup(emb_A, self.input_episodes)
        mem_C = tf.nn.embedding_lookup(emb_C, self.input_episodes)
        u = tf.nn.embedding_lookup(emb_B, self.input_querys)
        if self.PE:
            pos_emb = np.zeros((self.max_words, self.edim), np.float32)
            for j in range(1, self.max_words+1):
                for k in range(1, self.edim+1):
                    pos_emb[j-1, k-1] = (j - (self.edim+1)/2) * (j - (self.max_mem_size+1)/2)
            self.pos_emb = 1 + 4 * pos_emb / self.edim / self.max_mem_size

            # m_i = sum l_j * A_ij * x_ij
            mem_M = tf.multiply(mem_M, self.pos_emb)
            mem_C = tf.multiply(mem_C, self.pos_emb)
            u = tf.multiply(u, self.pos_emb)

        # to padded words be zeros and shouldn't be participated in backprob
        padding_mask_epi = tf.cast(tf.reshape(tf.sign(self.input_episodes), [self.batch_size, self.max_mem_size, self.max_words, 1]), tf.float32)
        padding_mask_time = tf.reduce_sum(padding_mask_epi, reduction_indices = 2)
        mem_M = tf.reduce_sum(tf.multiply(mem_M, padding_mask_epi), reduction_indices = 2)
        mem_C = tf.reduce_sum(tf.multiply(mem_C, padding_mask_epi), reduction_indices = 2)
        padding_mask_q = tf.cast(tf.reshape(tf.sign(self.input_querys), [self.batch_size, 1, self.max_words, 1]), tf.float32)
        u = tf.reduce_sum(tf.multiply(u, padding_mask_q), reduction_indices = 2)

        if self.TE:
            emb_T_A = self.get_variable("T_A1", shape = [self.max_mem_size, self.edim], stddev = self.init_std)
            emb_T_C = self.get_variable("T_C1", shape = [self.max_mem_size, self.edim], stddev = self.init_std)
            self._time = tf.stack([tf.range(0, self.max_mem_size)] * self.batch_size)
            T_M = tf.multiply(tf.nn.embedding_lookup(emb_T_A, self._time), padding_mask_time)
            T_C = tf.multiply(tf.nn.embedding_lookup(emb_T_C, self._time), padding_mask_time)
            mem_M = tf.add(mem_M, T_M)
            mem_C = tf.add(mem_C, T_C)
#if self.RN:

        for h in xrange(1, self.nhops+1):
            p = tf.reshape(tf.matmul(u, tf.transpose(mem_M, perm = [0, 2, 1])), [-1, self.max_mem_size])
            p = tf.reshape(tf.nn.softmax(p), [-1, self.max_mem_size, 1])
            o = tf.reduce_sum(tf.multiply(mem_C, p), reduction_indices = 1)
            o = tf.reshape(o, [-1, 1, self.edim])
            u = tf.add(o, u)
    
            if self.weight_tying == "Adj" and h < self.nhops:
                emb_C = self.get_variable("C%d"%(h+1), shape = [self.voca_size, self.edim], stddev = self.init_std)
                mem_M = mem_C
                mem_C = tf.nn.embedding_lookup(emb_C, self.input_episodes)

                if self.PE:
                    mem_C = tf.multiply(mem_C, self.pos_emb)

                mem_C = tf.reduce_sum(tf.multiply(mem_C, padding_mask_epi), reduction_indices = 2)
            
                if self.TE:
                    emb_T_C = self.get_variable("T_C%d"%(h+1), shape = [self.voca_size, self.edim], stddev = self.init_std)
                    T_C = tf.multiply(tf.nn.embedding_lookup(emb_T_C, self._time), padding_mask_time)
                    mem_C = tf.add(mem_C, T_C)
                
            elif self.weight_tying == "LW":
                if h > 1:
                    linear_H = self.get_variable("linear_H", shape = None, reuse = True)
                else:
                    linear_H = self.get_variable("linear_H", [self.edim, self.edim], stddev = self.init_std)
                u = tf.reshape(u, [-1, self.edim])
                u = tf.add(tf.reshape(tf.matmul(u, linear_H), [-1, 1, self.edim]), o)
            #else:
                #raise ValueError(" [!] Invaild weight tying type: %s"%self.weight_tying)

        if self.weight_tying == "Adj":
            self.W = tf.matrix_transpose(self.get_variable("C%d"%self.nhops, shape = None, reuse = True))
        else:
            self.W = self.get_variable("final_output", [self.edim, self.voca_size], stddev = self.init_std)

        self.u = tf.reshape(u, [-1, self.edim])
        self.logits = tf.matmul(self.u, self.W)
        self.preds = tf.nn.softmax(self.logits)
        print(" [*] Model build done!")

    def optimization(self):
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = self.labels))
        opt = tf.train.GradientDescentOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
        apply_grads = opt.apply_gradients(zip(grads, tvars))

        with tf.control_dependencies([apply_grads]):
            self.opt = tf.no_op()


    def run_epoch(self, dgen, desc = ""):
        iter_per_epoch = self.train_len / self.batch_size
        loss, acc = 0., 0.
        old_loss, counting = 999., 0
        pbar1 = trange(self.max_epoch)
        for epoch in pbar1:
            pbar1.set_description('Overall epoch')
            if ((epoch+1) % self.lr_anneal) == 0:
                self.lr.assign(self.lr * 0.5).eval()
            pbar2 = trange(iter_per_epoch)
            for step in pbar2:
                pbar2.set_description('Current epoch')
                querys, episodes, labels = dgen.next()
                _, loss, pred = self.sess.run([self.opt, self.loss, self.preds], 
                            feed_dict = { self.input_querys: querys,
                                        self.input_episodes: episodes,
                                        self.labels: labels 
                                        }
                            )
                acc = accuracy(pred, labels)
                pbar2.set_postfix(loss = loss, train_acc = "%.2f%%"%acc)

            test_acc, test_loss = self.test()
            pbar1.set_postfix(lr = self.sess.run(self.lr), loss = test_loss, test_acc = "%.2f%%"%test_acc)
            if old_loss > test_loss: 
                # no validation data, so we use test data as validation
                self.saver.save(self.sess,
                        os.path.join("checkpoints/", 
                                "memN2N_task%d_%s.model"%(self.task_id, desc)),
                        global_step = epoch)
                old_loss = test_loss
                counting = 0
            else:
                counting += 1
            # if you do validation check per step, indent all test code and change the ealy stopping criteria
            if counting > 50:
                print(" [#] early stopped...")
                break
        print("")

    def train(self):
        dgen = self.data_iteration(desc = "train")

        self.run_epoch(dgen)

        if self.LS:
            self.lr.assign(0.005).eval()
            self.RW = self.get_variable("LS_softmax", [self.edim, self.voca_size], stddev = self.    init_std)
            self.logits = tf.matmul(self.u, self.W)
            self.preds = tf.nn.softmax(self.logits)
            print(" [*] Training recommence")
            self.optimization()
            self.run_epoch(dgen, desc = 'LS')
            print(" [*] Learning end...")

    def test(self):
        dgen = self.data_iteration(desc = "test")
        iter_per_epoch = self.test_len / self.batch_size

        actual_label = []
        model_pred = []
        total_loss = []
        for step in range(iter_per_epoch):
            querys, episodes, labels = dgen.next()
            loss, pred = self.sess.run([self.loss, self.preds], 
                        feed_dict = { self.input_querys: querys,
                                    self.input_episodes: episodes,
                                    self.labels: labels 
                                    }
                        )
            total_loss.append(loss)
            actual_label.append(labels)
            model_pred.append(pred)
        total_loss = np.mean(total_loss)
        actual_label = np.concatenate(actual_label, axis = 0)
        model_pred = np.concatenate(model_pred, axis = 0)
        return accuracy(actual_label, model_pred), total_loss

    def data_iteration(self, desc = 'Train'):
        step = 0 # step for test data
        # in the paper, sentences are indexed in reverse order but no reversed in here
        while True:
            querys = np.zeros((self.batch_size, 1, self.max_words), np.int32)
            episodes = np.zeros((self.batch_size, self.max_mem_size, self.max_words), np.int32)
            labels = np.zeros((self.batch_size, self.voca_size), np.float32)

            batch_idxs = np.zeros((self.batch_size), np.int32)
            if desc.lower() == "train":
                batch_idxs = np.arange(self.train_len)[np.random.randint(self.train_len, size = self.batch_size)]
                data_story = self.train_story
                data_qstory = self.train_qstory
                data_questions = self.train_questions
            else:
                batch_offset = (step * self.batch_size) % self.test_len
                if batch_offset < self.batch_size and batch_offset != 0:
                    batch_idxs[:self.batch_size - batch_offset] = np.arange((step-1) * self.batch_size, self.test_len)
                    batch_idxs[self.batch_size - batch_offset:] = np.arange(batch_offset)
                else:
                    batch_idxs = np.arange(batch_offset, batch_offset + self.batch_size)
                data_story = self.test_story
                data_qstory = self.test_qstory
                data_questions = self.test_questions

            for idx, b in enumerate(batch_idxs):
                d = data_story[:, :(1 + self.train_questions[1, b]), self.train_questions[0, b]]
                offset = max(0, d.shape[1] - self.max_mem_size)
                querys[idx, 0, :] = data_qstory[:, b]
                # reverse order sentences
                episodes[idx, :d.shape[1], :d.shape[0]] = d[:, offset:].T[::-1]
                label = data_questions[2, b]
                labels[idx, :] = bow(self.dictionary, np.array([label]))
            yield querys, episodes, labels
