import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
from scipy import sparse
import math
import json
dmap = {
    'Math':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28, 27: 29, 28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40, 39: 41, 40: 42, 41: 43, 42: 44, 43: 45, 44: 46, 45: 47, 46: 48, 47: 49, 48: 50, 49: 51, 50: 52, 51: 53, 52: 54, 53: 55, 54: 56, 55: 57, 56: 58, 57: 59, 58: 60, 59: 61, 60: 62, 61: 63, 62: 64, 63: 65, 64: 66, 65: 67, 66: 68, 67: 69, 68: 70, 69: 71, 70: 72, 71: 73, 72: 74, 73: 75, 74: 76, 75: 77, 76: 78, 77: 79, 78: 80, 79: 81, 80: 82, 81: 83, 82: 84, 83: 85, 84: 86, 85: 87, 86: 88, 87: 89, 88: 90, 89: 91, 90: 92, 91: 93, 92: 94, 93: 95, 94: 96, 95: 97, 96: 98, 97: 99, 98: 100, 99: 101, 100: 102, 101: 103, 102: 105, 103: 106},
    'Lang': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 57, 54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65},
    # 'Lang': {0: 1, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 57, 54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65},
    'Chart':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 25, 24: 26},
    'Time':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27},
    'Mockito':{0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38},
    'Closure':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 29, 28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40, 39: 41, 40: 42, 41: 43, 42: 44, 43: 45, 44: 46, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 56, 54: 57, 55: 58, 56: 59, 57: 60, 58: 61, 59: 64, 60: 65, 61: 66, 62: 67, 63: 68, 64: 69, 65: 70, 66: 71, 67: 72, 68: 73, 69: 74, 70: 75, 71: 76, 72: 77, 73: 78, 74: 79, 75: 80, 76: 81, 77: 82, 78: 83, 79: 84, 80: 85, 81: 86, 82: 87, 83: 88, 84: 89, 85: 90, 86: 91, 87: 92, 88: 94, 89: 95, 90: 96, 91: 97, 92: 98, 93: 99, 94: 100, 95: 101, 96: 102, 97: 103, 98: 104, 99: 105, 100: 106, 101: 108, 102: 109, 103: 110, 104: 111, 105: 112, 106: 114, 107: 116, 108: 117, 109: 118, 110: 119, 111: 120, 112: 121, 113: 122, 114: 123, 115: 124, 116: 125, 117: 126, 118: 127, 119: 128, 120: 129, 121: 130, 122: 131, 123: 133, 124: 134, 125: 135, 126: 136, 127: 137, 128: 138, 129: 139, 130: 140, 131: 141, 132: 142, 133: 143, 134: 144, 135: 145, 136: 146, 137: 147, 138: 148, 139: 149, 140: 150, 141: 151, 142: 152, 143: 153, 144: 154, 145: 155, 146: 156, 147: 157, 148: 158, 149: 159, 150: 160, 151: 161, 152: 162, 153: 163, 154: 164, 155: 165, 156: 166, 157: 167, 158: 168, 159: 169, 160: 170, 161: 171, 162: 172, 163: 173, 164: 174, 165: 175, 166: 176},
    'Codec':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18},
    'Compress':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47},
    'Gson':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18},
    'Cli':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40},
    'Jsoup':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 15, 14: 16, 15: 17, 16: 19, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 56, 54: 57, 55: 58, 56: 59, 57: 60, 58: 61, 59: 62, 60: 63, 61: 64, 62: 65, 63: 66, 64: 67, 65: 68, 66: 70, 67: 71, 68: 72, 69: 73, 70: 74, 71: 75, 72: 76, 73: 77, 74: 78, 75: 79, 76: 80, 77: 81, 78: 82, 79: 83, 80: 84, 81: 85, 82: 86, 83: 87, 84: 88, 85: 89, 86: 90, 87: 91, 88: 92, 89: 93},
    'Csv':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16},
    'JacksonCore': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20},
    'JacksonXml': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    'Collections': {0: 25, 1: 26}
}
class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train", proj="Math", testid=0, lst=[]):
        self.train_path = proj + ".pkl"
        self.val_path = "ndev.txt"  # "validD.txt"
        self.test_path = "ntest.txt"
        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Voc['Method'] = len(self.Nl_Voc)
        self.Nl_Voc['Test'] = len(self.Nl_Voc)
        self.Nl_Voc['Line'] = len(self.Nl_Voc)
        self.Nl_Voc['RTest'] = len(self.Nl_Voc)
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.ids = []
        self.Nls = []
        if os.path.exists("nl_voc.pkl"):
        #    self.init_dic()
            self.Load_Voc()
        else:
            self.init_dic()
        print(self.Nl_Voc)
        if not os.path.exists(self.proj + 'data.pkl'):
            data = self.preProcessData(open(self.train_path, "rb"))
        else:
            data = pickle.load(open(self.proj + 'data.pkl', 'rb'))
        self.data = []
        if dataName == "train":
            for i in range(len(data)):
                #if testid == 0:
                #    self.data.append(data[i][testid + 1:])
                #elif testid == len(data[i]) - 1:
                #    self.data.append(data[i][0:testid] + data[i][testid + 1:])
                #else:
                tmp = []
                for j in range(len(data[i])):
                    if j in lst:
                        continue
                    tmp.append(data[i][j])#self.data.append(data[i][0:testid] + data[i][testid + 1:])
                self.data.append(tmp)
            #for i in range(len(data)):
            #    self.data = data[0:testid] + data[testid + 1:]
        elif dataName == 'test':
            #self.data = self.preProcessData(open('Lang.pkl', 'rb'))
            #self.ids = []
            testnum = 0#int(0.05 * len(data[0]))
            ids = []
            while len(ids) < testnum:
                rid = random.randint(0, len(data[0]) - 1)
                if rid == testid or rid in ids or rid == 51:#if rid >= testid * testnum and rid < testid * testnum + testnum or rid in ids:
                    continue
                ids.append(rid)
            self.ids = ids
            for i in range(len(data)):
                tmp = []
                for x in self.ids:
                    tmp.append(data[i][x])
                self.data.append(tmp)
        else:
            testnum = 1#int(0.1 * len(data[0]))
            ids = []
            for i in range(len(data)): 
                tmp = []
                for x in range(testnum * testid, testnum * testid + testnum):
                    if x < len(data[i]):
                        if i == 0:
                            ids.append(x)
                        tmp.append(data[i][x])
                self.data.append(tmp)
            self.ids = ids

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans
    def init_dic(self):
        print("initVoc")
        f = open(self.p + '.pkl', 'rb')
        data = pickle.load(f)
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for x in data:
            for s in x['methods']:
                s = s[:s.index('(')]
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
                print(Codes[-1])
            for s in x['ftest']:
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
        code_voc = VocabEntry.from_corpus(Codes, size=50000, freq_cutoff = 0)
        self.Code_Voc = code_voc.word2id
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def tokenize_for_bleu_eval(self, code):
        code = re.sub(r'([^A-Za-z0-9])', r' \1 ', code)
        #code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]
        return tokens
    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max(maxl, tmp)
            ans.append(int(100 * maxl / len(x)) + 1)
        return ans
    def getRes(self, codetoken, nltoken):
        ans = []
        for x in nltoken:
            if x == "<pad>":
                continue
            if x in codetoken and codetoken.index(x) < self.Code_Len and x != "(" and x != ")":
                ans.append(len(self.Nl_Voc) + codetoken.index(x))
            else:
                if x in self.Nl_Voc:
                    ans.append(self.Nl_Voc[x])
                else:
                    ans.append(1)
        for x in ans:
            if x >= len(self.Nl_Voc) + self.Code_Len:
                print(codetoken, nltoken)
                exit(0)
        return ans
    def preProcessData(self, dataFile):
        path_stacktrace = os.path.join('../FLocalization/stacktrace', self.proj)    
        lines = pickle.load(dataFile)#dataFile.readlines()
        Nodes = []
        Types = []
        LineNodes = []
        LineTypes = []
        LineMus = []
        Res = []
        inputText = []
        inputNlad = []
        maxl = 0
        maxl2 = 0
        error = 0
        error1 = 0
        error2 = 0
        correct = 0
        for k in range(len(lines)):
            x = lines[k]
            if os.path.exists(path_stacktrace + '/%d.json'%dmap[self.proj][k]):
                stack_info = json.load(open(path_stacktrace + '/%d.json'%dmap[self.proj][k]))
                if x['ftest'].keys() != stack_info.keys():
                    with open("problem_stack",'a') as f:
                        f.write("{} {} no!\n".format(k, dmap[self.proj][k]))
                        f.write(str(x['ftest'].keys()) + '\n')
                        f.write(str(stack_info.keys()) + '\n')
                    for error_trace in x['ftest'].keys():
                        if error_trace not in stack_info.keys():
                            error += 1
                        else:
                            correct += 1
                    
            nodes = []
            types = []
            res = []
            nladrow = []# = np.zeros([3200, 3200])
            nladcol = []
            nladval = []
            texta = []
            textb = []
            linenodes = []
            linetypes = []
            methodnum = len(x['methods'])
            rrdict = {}
            for s in x['methods']:
                rrdict[x['methods'][s]] = s[:s.index('(')]
            for i in range(methodnum):
                nodes.append('Method')
                if len(rrdict[i].split(":")) > 1:
                    tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:] + [rrdict[i].split(":")[1]]) 
                else:
                    tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:]) 
                ans = self.splitCamel(tokens)
                ans.remove('.')
                texta.append(ans)#(self.pad_seq(tmpids, 10))
                if i not in x['correctnum']:
                    types.append(1)
                else:
                    types.append(x['correctnum'][i] + 1)
                if i in x['ans']:
                    res.append(1)
                else:
                    res.append(0)
            rrdic = {}
            for s in x['ftest']:
                rrdic[x['ftest'][s]] = s
            for i in range(len(x['ftest'])):
                nodes.append('Test')
                types.append(0)
                if len(rrdic[i].split(":")) > 1:
                    tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:] + [rrdic[i].split(":")[1]])
                else:
                    tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:])
                ans = self.splitCamel(tokens)
                ans.remove('.')
                textb.append(ans)#(self.pad_seq(tmpids, 10))
            rrdic = {}
            for i in range(len(x['rtest'])):
                nodes.append('RTest')
                types.append(0)

            mus = []
            for i in range(len(x['lines'])):
                if i not in x['ltype']:
                    x['ltype'][i] = 'Empty'
                if x['ltype'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['ltype'][i]] = len(self.Nl_Voc)
                linenodes.append(x['ltype'][i])
                if i in x['lcorrectnum']:
                    linetypes.append(x['lcorrectnum'][i])
                else:
                    linetypes.append(1)
            '''for i in range(len(x['mutation'])):
                if x['mutation'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['mutation'][i]] = len(self.Nl_Voc)
                nodes.append(x['mutation'][i])
                types.append(0)'''
            maxl = max(maxl, len(nodes))
            maxl2 = max(maxl2, len(linenodes))
            ed = {}

            line2method = {}
            for e in x['edge2']:
                line2method[e[1]] = e[0]
                a = e[0]
                b = e[1] + self.Nl_Len#len(x['ftest']) + methodnum
                #nlad[a, b] = 1
                #nlad[b, a] = 1
                #assert(0)
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    print(a, b)
                    assert(0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    print(a, b)
                    assert(0)
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in x['edge10']:
                if e[0] not in line2method:
                    error1 += 1
                a = e[0] + self.Nl_Len
                b = e[1] + methodnum + len(x['ftest'])
                nladrow.append(a)
                nladcol.append(b)
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    pass
                    # print(e[0])
                    # print(a, b)
                    # assert(0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    pass
                    
                    # print(a, b)
                    # assert(0)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in x['edge']:
                if e[0] not in line2method:
                    error2 += 1
                a = e[0] + self.Nl_Len#+ len(x['ftest']) + methodnum
                b = e[1] + methodnum
                nladrow.append(a)
                nladcol.append(b)
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    print(e[0])
                    print(a, b)
                    assert(0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    print(a, b)
                    assert(0)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
                #nlad[a, b] = 1
                #nlad[b, a] = 1
           
            # for e in x['propagation_edges']:
            #     a = e[0]
            #     b = e[1]
                
            #     nladrow.append(a)
            #     nladcol.append(b)
            #     nladval.append(1)

            #     nladrow.append(b)
            #     nladcol.append(a)
            #     nladval.append(1)

            overlap = self.getoverlap(texta, textb)

            Nodes.append(self.pad_seq(self.Get_Em(nodes, self.Nl_Voc), self.Nl_Len))
            Types.append(self.pad_seq(types, self.Nl_Len))
            Res.append(self.pad_seq(res, self.Nl_Len))
            LineMus.append(self.pad_list(mus, self.Code_Len, 3))
            inputText.append(self.pad_seq(overlap, self.Nl_Len))
            LineNodes.append(self.pad_seq(self.Get_Em(linenodes, self.Nl_Voc), self.Code_Len))
            LineTypes.append(self.pad_seq(linetypes, self.Code_Len))
            row = {}
            col = {}
            for i  in range(len(nladrow)):
                if nladrow[i] not in row:
                    row[nladrow[i]] = 0
                row[nladrow[i]] += 1
                if nladcol[i] not in col:
                    col[nladcol[i]] = 0
                col[nladcol[i]] += 1
            for i in range(len(nladrow)):
                nladval[i] = 1 / math.sqrt(row[nladrow[i]]) * 1 / math.sqrt(col[nladcol[i]])
            nlad = sparse.coo_matrix((nladval, (nladrow, nladcol)), shape=(self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len))
            inputNlad.append(nlad)
        print("max1: %d max2: %d"%(maxl, maxl2))
        print("correct: %d error: %d"%(correct, error))
        print("error1: %d error2: %d"%(error1, error2))

        #assert(0)#assert(0)
        batchs = [Nodes, Types, inputNlad, Res, inputText, LineNodes, LineTypes, LineMus]
        self.data = batchs
        open(self.proj + "data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        #open('nl_voc.pkl', 'wb').write(pickle.dumps(self.Nl_Voc))
        return batchs

    def __getitem__(self, offset):
        ans = []
        if True:
            for i in range(len(self.data)):
                if i == 2:
                    ans.append(self.data[i][offset].toarray())
                else:
                    ans.append(np.array(self.data[i][offset]))
        else:
            for i in range(len(self.data)):
                if i == 4:
                    continue
                ans.append(np.array(self.data[i][offset]))
            negoffset = random.randint(0, len(self.data[0]) - 1)
            while negoffset == offset:
                negoffset = random.randint(0, len(self.data[0]) - 1)
            if self.dataName == "train":
                ans.append(np.array(self.data[2][negoffset]))
                ans.append(np.array(self.data[3][negoffset]))
        return ans
    def __len__(self):
        return len(self.data[0])
    def Get_Train(self, batch_size):
        data = self.data
        loaddata = data
        batch_nums = int(len(data[0]) / batch_size)
        if True:
            if self.dataName == 'train':
                shuffle = np.random.permutation(range(len(loaddata[0])))
            else:
                shuffle = np.arange(len(loaddata[0]))
            for i in range(batch_nums):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_size * i: batch_size * (i + 1)]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * i, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([batch_size, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                yield ans
            if batch_nums * batch_size < len(data[0]):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_nums * batch_size: ]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * batch_nums, len(data[0])):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([len(data[0]) - batch_size * batch_nums, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                yield ans
            
class node:
    def __init__(self, name):
        self.name = name
        self.father = None
        self.child = []
        self.id = -1