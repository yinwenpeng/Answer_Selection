import numpy

rootPath="/mounts/data/proj/wenpeng/Dataset/MicrosoftParaphrase/tokenized_msr/";
def Extract_Vocab():
    #get all words, then their word embeddings, discard unknown words
    readFile=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    dim=300
    word2vec=set()
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            word2vec.add(tokens[0])
    readFile.close()
    print 'word2vec vocab loaded over...'    
    files=['tokenized_train.txt', 'tokenized_test.txt']
    writeFile=open(rootPath+'vocab.txt', 'w')
    vocab={}
    count=0
    for file in files:
        readFile=open(rootPath+file, 'r')
        for line in readFile:
            tokens=line.strip().split('\t')
            for i in range(1,3):
                words=tokens[i].strip().lower().split() # lowercase makes more initialized words 
                for word in words:
                    key=vocab.get(word)
                    if key is None and word in word2vec:
                        count+=1
                        vocab[word]=count
                        writeFile.write(str(count)+'\t'+word+'\n')
                        
        readFile.close()
    writeFile.close()
    print 'total words: ', count
    


def transcate_word2vec_into_msr_vocab():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    dim=300
    word2vec={}
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            word2vec[tokens[0]]=map(float, tokens[1:])
    readFile.close()
    print 'word2vec loaded over...'
    readFile=open(rootPath+'vocab.txt', 'r')
    writeFile=open(rootPath+'vocab_embs_300d.txt', 'w')
    random_emb=list(numpy.random.uniform(-0.1,0.1,dim))
    for line in readFile:
        tokens=line.strip().split()
        emb=word2vec.get(tokens[1])
        if emb is None:
            emb=random_emb
        writeFile.write(tokens[1]+'\t'+' '.join(map(str, emb))+'\n')
    writeFile.close()
    readFile.close()
    print 'word2vec trancate over'

if __name__ == '__main__':
    #Extract_Vocab()
    transcate_word2vec_into_msr_vocab()
            