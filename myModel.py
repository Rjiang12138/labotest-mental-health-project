import sys, os

sys.path.append(os.getcwd())
import torch as th
from TextCNN import TextCNN
from Bert import BERTClass
from multi import MultiModalBERTClass

class myModel(th.nn.Module):
    def __init__(self, args):
        super(myModel, self).__init__()
        self.textcnn = TextCNN(args)
        self.bert = BERTClass(args)
        self.multi = MultiModalBERTClass(args)

    def forward(self, data):
        output2 = self.textcnn(data)
        output3 = self.bert(data)
        output4 = self.multi(data)
        return  output2, output3, output4
