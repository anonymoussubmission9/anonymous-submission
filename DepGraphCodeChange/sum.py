import pickle
import os
import sys
versionNum = {'Lang': 65, 'Time': 27, 'Chart': 26, 'Math': 106, 'Closure': 133, 'Mockito': 38,
                   'Cli': 40, 'Codec': 18, 'Csv': 16, 'CommonsJXPath': 14,
                   'JacksonCore': 20, 'JacksonDatabind': 39, 'JacksonXml': 5, 'Jsoup': 93, 'Compress':47, 'Gson':18, 'Collections':26}
proj = sys.argv[1]
seed = int(sys.argv[2])
lr = float(sys.argv[3])
batch_size = int(sys.argv[4])

t = {}
for i in range(0, versionNum[proj]):
    if not os.path.exists(proj + 'res%d_%d_%s_%s.pkl'%(i, seed, lr, batch_size)):
        continue
    p = pickle.load(open(proj + 'res%d_%d_%s_%s.pkl'%(i,seed, lr, batch_size), 'rb'))
    for x in p:
        t[x] = p[x]


    
open(proj + 'res_%d_%s_%s.pkl'%(seed,lr, batch_size), 'wb').write(pickle.dumps(t))
print(len(t))
