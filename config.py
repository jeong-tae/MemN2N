import numpy as np
import nltk

class GeneralConfig(object):
    
    def __init__(self, train_story, train_questions, dictionary):
        self.dictionary = dictionary
        self.batch_size = 64
        self.nhops = 3
        self.voca_size = len(self.dictionary)
        self.learning_rate = 0.01
        self.max_words = len(train_story)
        self.epi_size = min(50, train_story.shape[1])
        self.edim = 128

        self.init_std = 0.1
        self.max_iter = 20000

        self.train_range = np.array(range(train_questions.shape[1]))
        
        self.display_interval = 20
        self.test_interval = 1000

