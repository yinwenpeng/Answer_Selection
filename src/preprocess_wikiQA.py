import numpy
from itertools import izip


def filter_dev_test(devFile, testFile):
    #remove questions that have no answers
    #dev
    devread=open(devFile, 'r')
    devwrite=open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/dev_filtered.txt', 'w')
    single_question=[]
    flag=False
    pre_q=' '
    end=False
    empty_q=0
    for line in devread:
        parts=line.strip().split('\t')
        if parts[0]!=pre_q:#new question
            if pre_q!=' ':
                end=True
            #first write old
            if end is True and flag is True:
                for sent in single_question:
                    devwrite.write(sent+'\n')
            elif end is True and flag is False:
                empty_q+=1
            single_question=[]#empty this list
            end=False
            flag=False
            
        #remember no matter it is new or old
        single_question.append(line.strip())
        pre_q=parts[0]
        if parts[2]=='1':
            flag=True
    devwrite.close()
    devread.close()
    print empty_q, ' questions have no answers'

    #test
    testread=open(testFile, 'r')
    testwrite=open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/test_filtered.txt', 'w')
    single_question=[]
    flag=False
    pre_q=' '
    end=False
    empty_q=0
    for line in testread:
        parts=line.strip().split('\t')
        if parts[0]!=pre_q:#new question
            if pre_q!=' ':
                end=True
            #first write old
            if end is True and flag is True:
                for sent in single_question:
                    testwrite.write(sent+'\n')
            elif end is True and flag is False:
                empty_q+=1
            single_question=[]#empty this list
            end=False
            flag=False
            
        #remember no matter it is new or old
        single_question.append(line.strip())
        pre_q=parts[0]
        if parts[2]=='1':
            flag=True
    testwrite.close()
    testread.close()
    print empty_q, ' questions have no answers'   
    

def Extract_Vocab(path, train, dev, test):
    #consider all words, including unknown from word2vec, because some sentence 
    
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
    
    files=[train, dev, test]
    writeFile=open(path+'vocab_lower_in_word2vec.txt', 'w')
    vocab={}
    count=0
    for file in files:
        readFile=open(path+file, 'r')
        for line in readFile:
            tokens=line.strip().split('\t')
            for i in range(2):
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

def transcate_word2vec_into_wikiQA_vocab(rootPath):
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
    readFile=open(rootPath+'vocab_lower_in_word2vec.txt', 'r')
    writeFile=open(rootPath+'vocab_lower_in_word2vec_embs_300d.txt', 'w')
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

def compute_map_mrr(file, probs):
    #file
    testread=open(file, 'r')
    separate=[]
    labels=[]
    pre_q=' '
    line_no=0
    for line in testread:
        parts=line.strip().split('\t')
        if parts[0]!=pre_q:
            separate.append(line_no)
        labels.append(int(parts[2]))
        pre_q=parts[0]
        line_no+=1
    testread.close()
    separate.append(line_no)#the end of file
    #compute MAP, MRR
    question_no=len(separate)-1
    all_map=0.0
    all_mrr=0.0
    all_corr_answer=0
    for i in range(question_no):
        sub_labels=labels[separate[i]:separate[i+1]]
        sub_probs=probs[separate[i]:separate[i+1]]
        sub_dict = {k: v for k, v in izip(sub_probs, sub_labels)}
        sorted_probs=sorted(sub_probs, reverse = True) 
        map=0.0
        find=False
        corr_no=0
        #MAP
        for index, prob in enumerate(sorted_probs):
            if sub_dict[prob]==1:
                corr_no+=1
                all_corr_answer+=1
                map+=1.0*corr_no/(index+1)
                find=True
        #MRR
        for index, prob in enumerate(sorted_probs):
            if sub_dict[prob]==1:
                all_mrr+=1.0/(index+1)
                break # only consider the first correct answer              
        if find is False:
            print 'Did not find correct answers'
            exit(0)
        map=map/corr_no
        all_map+=map
    MAP=all_map/question_no
    MRR=all_mrr/question_no

    
    return MAP, MRR
    '''
    #compute MAP, MRR
    question_no=len(separate)-1
    all_map=0.0
    all_mrr=0.0
    all_corr_answer=0
    for i in range(question_no):
        sub_labels=labels[separate[i]:separate[i+1]]
        sub_probs=probs[separate[i]:separate[i+1]]
        all_map+=average_precision_score(numpy.array(sub_labels), numpy.array(sub_probs))  
        sub_dict = {k: v for k, v in izip(sub_probs, sub_labels)}
        sorted_probs=sorted(sub_probs, reverse = True) 

        find=False

        for index, prob in enumerate(sorted_probs):
            if sub_dict[prob]==1:
                all_corr_answer+=1
                all_mrr+=1.0/(index+1)
                find=True
                
        if find is False:
            print 'Did not find correct answers'
            exit(0)
    MAP=all_map/question_no
    MRR=all_mrr/all_corr_answer
    '''
              
        
    
    
if __name__ == '__main__':
    path='/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/'
    #filter_dev_test(path+'WikiQA-dev.txt', path+'WikiQA-test.txt')
    #Extract_Vocab(path, 'WikiQA-train.txt', 'dev_filtered.txt', 'test_filtered.txt')
    transcate_word2vec_into_wikiQA_vocab(path)
    #compute_map_mrr(path+'test_filtered.txt')



