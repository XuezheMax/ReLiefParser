__author__ = 'max'


class DependencyInstance(object):
    def __init__(self):
        self.words = []
        self.word_ids = []
        self.lemmas = []
        self.lemma_ids = []
        self.cpostags = []
        self.cpos_ids = []
        self.postags = []
        self.pos_ids = []
        self.heads = []
        self.types = []
        self.type_ids = []

    def length(self):
        return len(self.words)
