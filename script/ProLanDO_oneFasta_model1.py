################################################
#
#  Developed by Dr. Renzhi Cao
#  Pacific Lutheran University
#  Email: caora@plu.edu
#  Website: https://cs.plu.edu/~caora/
#
###############################################
import sys
import os
import torch


import torch.nn as nn
from torch.autograd import Variable
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
all_categories = ['DO00076', 'DO00002', 'DO00065', 'DO00050', 'DO00074', 'DO00011', 'DO00064', 'DO00025', 'DO00056', 'DO00028', 'DO00018', 'DO00063', 'DO00017', 'DO00015', 'DO00012', 'DO00047', 'DO00026', 'DO00001', 'DO00021', 'DO00071', 'DO00009', 'DO00008', 'DO00027', 'DO00035', 'DO00072', 'DO00052', 'DO00038', 'DO00005', 'DO00079', 'DO00019', 'DO00016', 'DO00053', 'DO00013', 'DO00014', 'DO00055', 'DO00078', 'DO00023', 'DO00003', 'DO00024', 'DO00077', 'DO00007', 'DO00029', 'DO00033', 'DO00051', 'DO00041', 'DO00022', 'DO00060', 'DO00030', 'DO00073', 'DO00037', 'DO00020', 'DO00006', 'DO00034', 'DO00046', 'DO00068', 'DO00058']

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor

def getScore(score):   # try to calculate the score from the output 
    if score < -1980*5:
       score = -1980*5 
    if score >0:
       score = 0
    scaled = round(1+score/10000.0,2)
    if scaled==1 and score<0:
       scaled=0.99 
    return scaled

def predict(inputfasta, rnn, output):
    #print('\n> %s' % input_line)
    fout = open(output,"w")
    fout.write("AUTHOR\tProLanDO\n")
    fout.write("MODEL\t1\n")
    fout.write("KEYWORDS\tde novo prediction, machine learning, natural language processing.\n") 
    
    # first read the fasta file
    fh = open(inputfasta,"r")
    targetName = None
    Seq = None
    for line in fh:
       if ">" in line:
          if Seq != None:
             # now evaluate and make predictions for previous targetName 
             output = evaluate(Variable(line_to_tensor(Seq)),rnn)
             # Get top N categories
             topv, topi = output.data.topk(len(all_categories), 1, True)
             if len(topv[0]) == 0:
                # we only make one prediction with low score 
                fout.write(targetName+"\t"+"DO:DO00076"+"\t0.01\n")
             else:
                removeDup = {}
                myScore = 1.00  # start from here, and deduct 0.02 each time 
                for i in range(len(topv[0])):
                    value = topv[0][i]
                    category_index = topi[0][i]
                    if category_index > len(all_categories) or category_index<0:
                        category_index = 0
                    if category_index in removeDup:
                        continue   # remove the duplicates 
                    removeDup[category_index] = 1 
                    fout.write(targetName+"\t"+"DO:"+all_categories[category_index]+"\t"+str(round(myScore,2))+"\n")
                    myScore-=0.02
                    if myScore < 0.02:
                        break
             Seq = None # empty the sequence for new prediction
          targetName = line.split()[0][1:]     # this is the new target name 
       else:
          if Seq == None:
             Seq = line.strip()
          else:
             Seq = Seq+line.strip()  
    if targetName!=None and Seq!=None:   # make prediction for the last sequence
       output = evaluate(Variable(line_to_tensor(Seq)),rnn)
       topv, topi = output.data.topk(len(all_categories), 1, True)
       if len(topv[0]) == 0:
          fout.write(targetName+"\t"+"DO:DO00076"+"\t0.01\n")
       else:
          removeDup = {}
          myScore = 1.00
          for i in range(len(topv[0])):
             value = topv[0][i]
             category_index = topi[0][i]
             if category_index > len(all_categories) or category_index<0:
                category_index = 0
             if category_index in removeDup:
                continue
             removeDup[category_index] = 1
             fout.write(targetName+"\t"+"DO:"+all_categories[category_index]+"\t"+str(round(myScore,2))+"\n")
             myScore-=0.02
             if myScore<0.02:
                break    # get enough prediction
    fh.close()
 
    fout.write("END\n")
    fout.close()

def evaluate(line_tensor, rnn):
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output


if len(sys.argv)<4:
   print("This script would take a fasta file as input and make predictions for DO, but you may need to change code inside for output server names.")
   print("This is for prediction, use: \npython "+sys.argv[0]+" address_model addr_fasta addr_output")
   print("python "+sys.argv[0]+" ../model/training_DO_RNN_512_0.009000000000000001.model ../data/TargetFiles/sp_species.287.tfa ../result/CaoLab_1_287_do.txt")
   sys.exit(0)


modelPath = sys.argv[1]
model = torch.load(modelPath, map_location='cpu')   # torch.load(TrainedModel, map_location='cpu'
model.eval()

inputFasta = sys.argv[2]
addOut = sys.argv[3]
predict(inputFasta, model, addOut)
