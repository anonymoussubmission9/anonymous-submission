import pickle
with open('/home/r_mdnakh/grace-org/Code/Default/Closure_Original.pkl', 'rb') as f:
    data = pickle.load(f)
    dict = {}
    count = 0
    for d in data:
        # methods = d['methods']
        # edge10 = d['edge10']
        # edge = d['edge']
        # edge2 = d['edge2']
        # rtest = d['rtest']
        # ftest = d['ftest']
        # methods = d['methods']
        # lines = d['lines']
        # ans = d['ans']
        proj = d['proj']
        # modifcation = d['modification']
        # print(modifcation)
        # print('-'*20)
        # print('Project:', proj)
        # dict[count] = int(proj[4:])
        dict[count] = int(proj[7:])
        count += 1
 
        # print(methods)
   
    print(dict)
    # print(time_dict)