import torch
from torch import optim
from Dataset import SumDataset
import os
from tqdm import tqdm
from Model import *
import numpy as np
import time
#from annoy import AnnoyIndex
from nltk import word_tokenize
import pickle
from ScheduledOptim import ScheduledOptim
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import random
import sys
#import wandb
#wandb.init(project="codesum")
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

NlLen_map = {"Time":3900, "Math":3000, "Lang":350, "Chart": 2350, "Mockito":1370, "Codec":160, "Compress":1000, "Gson":1500, "Cli":1000, "Jsoup":2000, "Csv":1000, "JacksonCore":1000, 'JacksonXml':500, 'Collections':500}
CodeLen_map = {"Time":1300, "Math":1000, "Lang":350, "Chart":5250, "Mockito":280, "Codec":190, "Compress":1500, "Gson":1500, "Cli":1000, "Jsoup":2000, "Csv":1000, "JacksonCore":1000, 'JacksonXml':500, 'Collections':500}
args = dotdict({
    'NlLen':NlLen_map[sys.argv[2]],
    'CodeLen':CodeLen_map[sys.argv[2]],
    'SentenceLen':10,
    'batch_size':60,
    'embedding_size':32,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'seed':0,
    'lr':1e-3
})
os.environ['PYTHONHASHSEED'] = str(args.seed)

def save_model(model, dirs = "checkpointcodeSearch"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + '/best_model.ckpt')


def load_model(model, dirs="checkpointcodeSearch"):
    assert os.path.exists(dirs + '/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + '/best_model.ckpt'))

use_cuda = torch.cuda.is_available()
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor

def train(t = 5, p='Math'):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed + t)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev_set = SumDataset(args, "test", p, testid=t)
    val_set = SumDataset(args, "val", p, testid=t)
    data = pickle.load(open(p + '.pkl', 'rb'))
    dev_data = pickle.load(open(p + '.pkl', 'rb'))
    train_set = SumDataset(args, "train", testid=t, proj=p, lst=dev_set.ids + val_set.ids)
    numt = len(train_set.data[0])
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)

    print(dev_set.ids)
    model = NlEncoder(args)
    if use_cuda:
        print('using GPU')
        model = model.cuda()
    maxl = 1e9
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)
    maxAcc = 0
    minloss = 1e9
    rdic = {}
    brest = []
    bans = []
    batchn = []
    each_epoch_pred = {}
    for x in dev_set.Nl_Voc:
      rdic[dev_set.Nl_Voc[x]] = x
    
    # Training time starts here
    train_start_time = time.time()
    cumulative_test_time = 0  # To hold the cumulative testing time across all epochs
    for epoch in range(15):
        index = 0
        for dBatch in tqdm(train_set.Get_Train(args.batch_size)):
            if index == 0:
                accs = []
                loss = []
                model = model.eval()
                # Testing time starts here
                test_start_time = time.time()
                score2 = []
                for k, devBatch in tqdm(enumerate(val_set.Get_Train(len(val_set)))):
                        for i in range(len(devBatch)):
                            devBatch[i] = gVar(devBatch[i])
                        with torch.no_grad():
                            l, pre, _ = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], devBatch[6], devBatch[7], devBatch[8], devBatch[9])
                            resmask = torch.eq(devBatch[0], 2)
                            s = -pre#-pre[:, :, 1]
                            s = s.masked_fill(resmask == 0, 1e9)
                            pred = s.argsort(dim=-1)
                            pred = pred.data.cpu().numpy()
                            alst = []
                            score_dict = {}

                            for k in range(len(pred)): 
                                datat = data[val_set.ids[k]]
                                maxn = 1e9
                                lst = pred[k].tolist()[:resmask.sum(dim=-1)[k].item()]#score = np.sum(loss) / numt
                                for pos in lst:
                                    score_dict[pos] = s[k, pos].item()
                                #bans = lst
                                for x in datat['ans']:
                                    i = lst.index(x)
                                    maxn = min(maxn, i)
                                score2.append(maxn)

                test_end_time = time.time()  # Testing time ends here
                total_test_time = test_end_time - test_start_time
                cumulative_test_time += total_test_time
                # print(f'Total testing time for epoch {epoch}: {total_test_time} seconds')
                each_epoch_pred[epoch] = lst
                each_epoch_pred[str(epoch)+'_pred'] = score_dict
                score = score2[0]
                print('curr accuracy is ' + str(score) + "," + str(score2))
                if score2[0] == 0:
                    batchn.append(epoch)
                    

                if  maxl >= score:
                    brest = score2
                    bans = lst
                    maxl = score
                    print("find better score " + str(score) + "," + str(score2))
                    #save_model(model)
                    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
                model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5], dBatch[6], dBatch[7], dBatch[8], dBatch[9])
            print(loss.mean().item())
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()

            optimizer.step_and_update_lr()
            index += 1
    train_end_time = time.time()  # Training time ends here
    total_train_time = train_end_time - train_start_time

    print(f"TIMING_INFO: Training Time: {total_train_time}, Testing Time: {cumulative_test_time}")
    with open(f'{p}_timing_data.txt', 'a') as f:  # 'a' mode is for appending to the file
        f.write(f"TIMING_INFO: Training Time: {total_train_time}, Testing Time: {cumulative_test_time}\n")

    return brest, bans, batchn, each_epoch_pred



if __name__ == "__main__":
    args.lr = float(sys.argv[3])
    args.seed = int(sys.argv[4])
    args.batch_size = int(sys.argv[5])
    np.set_printoptions(threshold=sys.maxsize)
    res = {}    
    p = sys.argv[2]
    res[int(sys.argv[1])] = train(int(sys.argv[1]), p)
    open('%sres%d_%d_%s_%s.pkl'%(p, int(sys.argv[1]), args.seed, args.lr, args.batch_size), 'wb').write(pickle.dumps(res))



