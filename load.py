import numpy as np #ACH平均+各胺基酸出現機率
with open('/home/mlb2017/res/phosphosite/train_data') as f:#train_data
    data = f.readlines()
data = [x.strip() for x in data] 
with open('/home/mlb2017/res/phosphosite/validation_data') as f:#valid_data
    data2 = f.readlines()
data2 = [x.strip() for x in data2] 
def newfeature(data):
    seqlist = []
    for site in data:
        site = site.split(' ')
        seqlist.append(site[2])
    #print(seqlist)
    hyd={'A':0.62,'C':0.29,'D':-0.90,'E':-0.74,'F':1.19,'G':0.48,'H':-0.40,'I':1.38,'K':-1.50,'L':1.06,'M':0.64,'N':-0.78,'P':0.12,'Q':-0.85,'R':-2.53,'S':-0.18,'T':-0.05,'V':1.08,'W':0.81,'Y':0.26}  
    acid=['A', 'C','D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    ACH_feature = []
    for seq in seqlist:
        each=[]
        start=8
        for k in range(0,20):
            count=seq.count(acid[k])
            each.append(count)
        for i in range(3,21,2):
            sum=0
            for j in range(0,i):
                sum+=hyd[seq[start+j]]
            sum/=i
            each.append(sum)
            start-=1  
        ACH_feature.append(each)
    feature = np.array(ACH_feature)
    return feature
train_f=newfeature(data)
valid_f=newfeature(data2)
#print(train_f[0])
#print(valid_f[0])
X = np.load('/home/mlb2017/res/phosphosite/trainX.npy')
A = np.load('/home/mlb2017/res/phosphosite/validationX.npy')
X = np.hstack((train_f, X)) # you can use hstack to concate your new feature on old feature
A = np.hstack((valid_f, A)) 
Y = np.load('/home/mlb2017/res/phosphosite/trainY.npy')
B = np.load('/home/mlb2017/res/phosphosite/validationY.npy')
np.save('train_X', X)
np.save('train_Y', Y)
np.save('validationX', A)
np.save('validationY', B)
