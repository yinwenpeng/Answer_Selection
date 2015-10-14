import random
import numpy

def word2key(word):
    #map each word into a key, like "haha" into "aahh".
    return ''.join(sorted(map(lambda x:x, word)))

def anagram(dicFile):

    readFile=open(dicFile,'r')
    key2cluster={} #put all words with the same key into a cluster
    vocab=[] #used for iteration in output
    for line in readFile:
        word=line.strip()
        vocab.append(word)
        key=word2key(word)
        cluster=key2cluster.get(key)
        if cluster is None:
            cluster=set()
        cluster.add(word)
        key2cluster[key]=cluster
    readFile.close()
    #print anagrams for each word
    writeFile=open('word2anagram.txt','w')
    for word in vocab:
        key=word2key(word)
        cluster=key2cluster.get(key)
        cluster.remove(word)# only print anagrams
        writeFile.write(word)
        if len(cluster)>0:         
            for word in cluster:
                writeFile.write(' '+word)
        writeFile.write('\n')            
    writeFile.close()
    
    
    
    
    
def test():

    map={}
    for i in range(100):   
        map[i]=i**1.2
    haha=random.sample(map.items(), 3) 
    for (key, value) in haha:
        print key, value

def transcate_word2vec_into_ibmvocab():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/word2vec_50d_Heike.txt', 'r')
    dim=50
    word2vec={}
    line_count=0
    for line in readFile:
        line_count+=1
        if line_count==1:
            continue
        else:
            tokens=line.strip().split()
            word2vec[tokens[0]]=map(float, tokens[1:])
    readFile.close()
    print 'word2vec loaded over...'
    readFile=open('/mounts/data/proj/wenpeng/Dataset/insuranceQA/vocabulary', 'r')
    writeFile=open('/mounts/data/proj/wenpeng/Dataset/insuranceQA/vocab_embs.txt', 'w')
    random_emb=list(numpy.random.uniform(-0.1,0.1,dim))
    for line in readFile:
        tokens=line.strip().split()
        emb=word2vec.get(tokens[1])
        if emb is None:
            emb=random_emb
        writeFile.write(tokens[1]+'\t')
        for value in emb:
            writeFile.write(str(value)+' ')
        writeFile.write('\n')
    writeFile.close()
    readFile.close()
    print 'word2vec trancate over'
            
    
    
if __name__ == '__main__':
    print numpy.random.uniform(-0.1,0.1,50)

