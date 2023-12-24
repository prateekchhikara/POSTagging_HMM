import json
import numpy as np

class Greedy:
    
    def __init__(self, testData):
        
        with open('hmm.json') as json_file:
            data = json.load(json_file)
        print(data.keys())
        self.emissionDict = data["emission"]
        self.transitionDict = data["transition"]
        self.testData = testData + ["\n"]
        self.sentences = []
        self.allSentTokens = []
        self.finalPredictedPOS = []
        
        with open("vocab.txt", "r") as f:
            myFile = f.readlines()
            
        self.myVocab = [i.split("\t")[0] for i in myFile]
        
        self.frequencyPOSNormalized = {'NNP': 0.09605139815479748, ',': 0.05095960398861961, 'CD': 0.03823724502381879, \
            'NNS': 0.06343527812344109, 'JJ': 0.06462484719245254, 'MD': 0.010346509957844304, 'VB': 0.027945553917081006, \
            'DT': 0.08636709991831991, 'NN': 0.13982534714037465, 'IN': 0.10389049386302962, '.': 0.041534050729364815, \
            'VBZ': 0.023004182678339428, 'VBG': 0.015730817513526552, 'CC': 0.025016034513948657, 'VBD': 0.03103733711948865, \
            'VBN': 0.021192967837780057, 'RB': 0.0324757837725237, 'TO': 0.023529347271939876, 'PRP': 0.018381857153037785, \
            'RBR': 0.00183643151206837, 'WDT': 0.004598205230814773, 'VBP': 0.013513943174778944, 'RP': 0.002757388210657881, \
            'PRP$': 0.008758956029799527, 'JJS': 0.002046935900317401, 'POS': 0.009082387251327987, '``': 0.007435628964088171, \
            'EX': 0.0009132820594345984, "''": 0.007260208640547311, 'WP': 0.002505221495567896, ':': 0.005131044463570132, \
            'JJR': 0.003479900668241795, 'WRB': 0.002247572895367259, '$': 0.007605567402518378, 'NNPS': 0.0027464244404365773, \
            'WP$': 0.00018199858567364145, '-LRB-': 0.001430772013880133, '-RRB-': 0.001448314046234219, 'PDT': 0.0003650935483694133,\
            'RBS': 0.00047692400462671106, 'FW': 0.0002455884529572029, 'UH': 9.538480092534221e-05, 'SYM': 6.030073621717036e-05,\
            'LS': 5.15297200401274e-05, '#': 0.00013923988181055701}
        self.POSlist = ['NNP', ',', 'CD', 'NNS', 'JJ', 'MD', 'VB',\
            'DT', 'NN', 'IN', '.', 'VBZ', 'VBG', 'CC', 'VBD', \
            'VBN', 'RB', 'TO', 'PRP', 'RBR', 'WDT', 'VBP', \
            'RP', 'PRP$', 'JJS', 'POS', '``', 'EX', "''", 'WP', \
            ':', 'JJR', 'WRB', '$', 'NNPS', 'WP$', '-LRB-', '-RRB-', \
            'PDT', 'RBS', 'FW', 'UH', 'SYM', 'LS', '#']
        
    def createSentences(self,):
        sent = []
        for d in self.testData:
            s = d.split("\t")
            if len(s) < 2:
                self.sentences.append(sent)
                self.allSentTokens.extend(sent)
                sent = []
                continue
            
            sent.append(s[1][:-1])
            
    def predict(self):
        for sent in self.sentences:
            pred = self.predictPOS(sent)
            self.finalPredictedPOS.extend(pred)
        
    def predictPOS(self, sent):
        Viterbi = []
        posPredicted = []

        valMatrix = [0]*len(self.POSlist)
        for i in range(len(valMatrix)):
            if f"{self.POSlist[i]}|{sent[0]}" not in self.emissionDict:
                valMatrix[i] = 0.0000001
            else:
                valMatrix[i] = self.emissionDict[f"{self.POSlist[i]}|{sent[0]}"] * self.frequencyPOSNormalized[self.POSlist[i]]
                
        Viterbi.append(valMatrix)
        z = valMatrix.index(max(valMatrix))
        posPredicted.append(self.POSlist[z])

        for i in range(1, len(sent)):
            w = sent[i]
            if w not in self.myVocab:
                w = "<unk>"
            valMatrix = [0]*len(self.POSlist)
            for j in range(len(self.POSlist)):
                temp = []
                for k in range(len(self.POSlist)):
                    if f"{self.POSlist[j]}|{w}" not in self.emissionDict or f"{self.POSlist[k]}|{self.POSlist[j]}" not in self.transitionDict:
                        temp.append(np.float64(0.0000001))
                    else:
                        temp.append(np.float64(self.emissionDict[f"{self.POSlist[j]}|{w}"] * self.transitionDict[f"{self.POSlist[k]}|{self.POSlist[j]}"] * Viterbi[-1][k] + 0.0001))
                valMatrix[j] = max(temp)
            Viterbi.append(valMatrix)
        
            z = valMatrix.index(max(valMatrix))
            posPredicted.append(self.POSlist[z])
                
            
        return posPredicted
        
        
if __name__ == "__main__":
    # read test data
    with open("data/test") as f:
        data = f.readlines()

    obj = Greedy(data)
    obj.createSentences()
    obj.predict()
    # print(obj.myVocab)
    # print(len(obj.finalPredictedPOS))
    # print(len(obj.allSentTokens))
    
    index = 0
    with open("viterbi.out", "w+") as f:
        for i in range(len(obj.sentences)):
            sent = obj.sentences[i]
            idx = 1
            for token in sent:
                f.write(f"{idx}\t{token}\t{obj.finalPredictedPOS[index]}\n")
                idx += 1
                index += 1
            f.write("\n")

    