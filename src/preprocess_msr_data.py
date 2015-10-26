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

def putAllMtTogether():
    pathroot='/mounts/data/proj/wenpeng/Dataset/paraphraseMT/'
    train_files=[pathroot+'badger/output_trainparaphrase/Badger-seg.scr', pathroot+'BLEU&NIST/Paraphrase/result_train/BLEU-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_train/NIST-seg.scr', pathroot+'maxsim-v1.01/paraphrase/train.score',
                 pathroot+'METROE/meteor-1.4/paraphrase/train_score_pure.txt', pathroot+'SEPIA/SEPIA_PKG_0.2/paraphraseTrainResult/system01-seg.scr',
                 pathroot+'TER/tercom-0.7.25/paraphrase/train_score.ter', pathroot+'TERp/terp.v1/paraphrase/output_traindata/terpa.simple.system01.seg.scr']
    
    test_files=[pathroot+'badger/output_testparaphrase/Badger-seg.scr', pathroot+'BLEU&NIST/Paraphrase/result_test/BLEU-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_test/NIST-seg.scr', pathroot+'maxsim-v1.01/paraphrase/test.score',
                 pathroot+'METROE/meteor-1.4/paraphrase/test_score_pure.txt', pathroot+'SEPIA/SEPIA_PKG_0.2/paraphraseTestResult/system01-seg.scr',
                 pathroot+'TER/tercom-0.7.25/paraphrase/test_score.ter', pathroot+'TERp/terp.v1/paraphrase/output_testdata/terpa.simple.system01.seg.scr']

    posi=[4, 4,4,1, 3,4, 3,4]
    
    train_write=open(pathroot+'concate_8mt_train.txt', 'w')
    scores=[]
    for i in range(8):
        read_file=open(train_files[i], 'r')
        list_values=[]
        for line in read_file:
            tokens=line.strip().split()
            list_values.append(tokens[posi[i]])
        read_file.close()
        scores.append(list_values)
    values_matrix=numpy.array(scores)
    col=values_matrix.shape[1]
    for j in range(col):
        for i in range(8):
            train_write.write(values_matrix[i,j]+'\t')
        train_write.write('\n')
    train_write.close()
    #test
    test_write=open(pathroot+'concate_8mt_test.txt', 'w')
    scores=[]
    for i in range(8):
        read_file=open(test_files[i], 'r')
        list_values=[]
        for line in read_file:
            tokens=line.strip().split()
            list_values.append(tokens[posi[i]])
        read_file.close()
        scores.append(list_values)
    values_matrix=numpy.array(scores)
    col=values_matrix.shape[1]
    for j in range(col):
        for i in range(8):
            test_write.write(values_matrix[i,j]+'\t')
        test_write.write('\n')
    test_write.close()
    print 'finished'
    

if __name__ == '__main__':
    #Extract_Vocab()
    #transcate_word2vec_into_msr_vocab()
    putAllMtTogether()
            