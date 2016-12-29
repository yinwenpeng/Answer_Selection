import numpy
from string import digits

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
    random_emb=list(numpy.random.uniform(-0.01,0.01,dim))
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
    train_files=[pathroot+'badger/output_trainparaphrase/Badger-seg.scr', pathroot+'BLEU&NIST/Paraphrase/result_train/BLEU1-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_train/BLEU2-seg.scr',pathroot+'BLEU&NIST/Paraphrase/result_train/BLEU3-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_train/BLEU4-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_train/NIST1-seg.scr',pathroot+'BLEU&NIST/Paraphrase/result_train/NIST2-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_train/NIST3-seg.scr',pathroot+'BLEU&NIST/Paraphrase/result_train/NIST4-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_train/NIST5-seg.scr',
                  pathroot+'maxsim-v1.01/paraphrase/train.score',
                 pathroot+'METROE/meteor-1.4/paraphrase/train_score_pure.txt', pathroot+'SEPIA/SEPIA_PKG_0.2/paraphraseTrainResult/system01-seg.scr',
                 pathroot+'TER/tercom-0.7.25/paraphrase/train_score.ter', pathroot+'TERp/terp.v1/paraphrase/output_traindata/terpa.simple.system01.seg.scr']
    
    test_files=[pathroot+'badger/output_testparaphrase/Badger-seg.scr', pathroot+'BLEU&NIST/Paraphrase/result_test/BLEU1-seg.scr',
                pathroot+'BLEU&NIST/Paraphrase/result_test/BLEU2-seg.scr',pathroot+'BLEU&NIST/Paraphrase/result_test/BLEU3-seg.scr',
                pathroot+'BLEU&NIST/Paraphrase/result_test/BLEU4-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_test/NIST1-seg.scr',pathroot+'BLEU&NIST/Paraphrase/result_test/NIST2-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_test/NIST3-seg.scr',pathroot+'BLEU&NIST/Paraphrase/result_test/NIST4-seg.scr',
                 pathroot+'BLEU&NIST/Paraphrase/result_test/NIST5-seg.scr',
                  pathroot+'maxsim-v1.01/paraphrase/test.score',
                 pathroot+'METROE/meteor-1.4/paraphrase/test_score_pure.txt', pathroot+'SEPIA/SEPIA_PKG_0.2/paraphraseTestResult/system01-seg.scr',
                 pathroot+'TER/tercom-0.7.25/paraphrase/test_score.ter', pathroot+'TERp/terp.v1/paraphrase/output_testdata/terpa.simple.system01.seg.scr']

    posi=[4, 4,4,4,4,  4,4,4,4,4, 1, 3,4, 3,4]
    
    train_write=open(pathroot+'concate_15mt_train.txt', 'w')
    scores=[]
    for i in range(15):
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
        for i in range(15):
            train_write.write(values_matrix[i,j]+'\t')
        train_write.write('\n')
    train_write.close()
    #test
    test_write=open(pathroot+'concate_15mt_test.txt', 'w')
    scores=[]
    for i in range(15):
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
        for i in range(15):
            test_write.write(values_matrix[i,j]+'\t')
        test_write.write('\n')
    test_write.close()
    print 'finished'
def two_word_matching_methods(path, trainfile, testfile):
    stop_word_list=open(path+'short-stopwords.txt', 'r')
    stop_words=set()
    
    for line in stop_word_list:
        word=line.strip()
        stop_words.add(word)
    stop_word_list.close()
    print 'totally ', len(stop_words), ' stop words'
    #word 2 idf
    word2idf={}
    for file in [trainfile, testfile]:
        read_file=open(path+file, 'r')
        for line in read_file:
            parts=line.strip().split('\t')
            for i in [1,2]:
                sent2set=set(parts[i].split())# do not consider repetition
                for word in sent2set:
                    if word not in stop_words:
                        count=word2idf.get(word,0)
                        word2idf[word]=count+1
        read_file.close()
        
    WC_train=[]
    WWC_train=[]
    #train file
    read_train=open(path+trainfile, 'r')
    #write_train=open(path+'train_word_matching_scores_normalized.txt','w')
    for line in read_train:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[1].split()
        answer=parts[2].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        WC_train.append(WC)
        WWC_train.append(WWC)
        #change question and answer
        WC=0
        WWC=0
        question=parts[2].split()
        answer=parts[1].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        WC_train.append(WC)
        WWC_train.append(WWC)
        #write_train.write(str(WC)+' '+str(WWC)+'\n')
    #write_train.close()
    read_train.close()
    
    #test file
    WC_test=[]
    WWC_test=[]
    read_test=open(path+testfile, 'r')
    #write_test=open(path+'test_word_matching_scores.txt','w')
    for line in read_test:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[1].split()
        answer=parts[2].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        WC_test.append(WC)
        WWC_test.append(WWC)
        #write_test.write(str(WC)+' '+str(WWC)+'\n')
    #write_test.close()
    read_test.close()   
    WC_overall=WC_train+WC_test
    max_WC=numpy.max(WC_overall)          
    min_WC=numpy.min(WC_overall)
    
    write_train=open(path+'train_word_matching_scores.txt','w')
    for index,wc in enumerate(WC_train):
        #wc=(wc-min_WC)*1.0/(max_WC-min_WC) # normalize
        write_train.write(str(wc)+' '+str(WWC_train[index])+'\n')
    write_train.close()
    write_test=open(path+'test_word_matching_scores.txt','w')
    for index,wc in enumerate(WC_test):
        #wc=(wc-min_WC)*1.0/(max_WC-min_WC)
        write_test.write(str(wc)+' '+str(WWC_test[index])+'\n')
    write_test.close()    
    
    print 'two word matching values generated'    

def Number_Overlap_Features(path, trainfile, testfile):

    #train file
    write_train=open(path+'train_number_matching_scores.txt','w')
    read_train=open(path+trainfile, 'r')
    #write_train=open(path+'train_word_matching_scores_normalized.txt','w')
    for line in read_train:
        parts=line.strip().split('\t')
        question=parts[1].split()
        no_set_q=set()
        for word in question:
            if containsnumbers(word):
                no_set_q.add(word)
        no_set_a=set()
        answer=parts[2].split()
        for word in answer:
            if containsnumbers(word):
                no_set_a.add(word)      
        feature_1=0      
        if no_set_q==no_set_a:
            feature_1=1
        feature_2=0
        if no_set_q < no_set_a or no_set_q > no_set_a:
            feature_2=1
        feature_3=0
        if len(no_set_q & no_set_a)>0:
            feature_3=1
        write_train.write(str(feature_1)+' '+str(feature_2)+' '+str(feature_3)+'\n')    
        write_train.write(str(feature_1)+' '+str(feature_2)+' '+str(feature_3)+'\n')  #repeat once  
    read_train.close()
    write_train.close()
    #test file
    write_test=open(path+'test_number_matching_scores.txt','w')
    read_test=open(path+testfile, 'r')
    #write_train=open(path+'train_word_matching_scores_normalized.txt','w')
    for line in read_test:
        parts=line.strip().split('\t')
        question=parts[1].split()
        no_set_q=set()
        for word in question:
            if containsnumbers(word):
                no_set_q.add(word)
        no_set_a=set()
        answer=parts[2].split()
        for word in answer:
            if containsnumbers(word):
                no_set_a.add(word)      
        feature_1=0      
        if no_set_q==no_set_a:
            feature_1=1
        feature_2=0
        if no_set_q < no_set_a or no_set_q > no_set_a:
            feature_2=1
        feature_3=0
        if len(no_set_q & no_set_a)>0:
            feature_3=1
        write_test.write(str(feature_1)+' '+str(feature_2)+' '+str(feature_3)+'\n')     
    read_test.close()
    write_test.close()
    
    print 'two number matching values generated'  
    
def containsnumbers(value):
    return any(char in digits for char in value)
if __name__ == '__main__':
    #Extract_Vocab()
    #transcate_word2vec_into_msr_vocab()
    #putAllMtTogether()
    #two_word_matching_methods('/mounts/data/proj/wenpeng/Dataset/MicrosoftParaphrase/tokenized_msr/', 'tokenized_train.txt', 'tokenized_test.txt')
    Number_Overlap_Features('/mounts/data/proj/wenpeng/Dataset/MicrosoftParaphrase/tokenized_msr/', 'tokenized_train.txt', 'tokenized_test.txt')
            