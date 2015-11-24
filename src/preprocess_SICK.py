import numpy
from itertools import izip
from xml.sax.saxutils import escape
from nltk.tokenize import TreebankWordTokenizer

def extract_pairs(path, inputfile):
    read_file=open(path+inputfile, 'r')
    train_file=open(path+'train.txt', 'w')
    test_file=open(path+'test.txt', 'w')
    dev_file=open(path+'dev.txt', 'w')
    train_no=0
    test_no=0
    dev_no=0
    remain_no=0
    line_no=0
    for line in read_file:
        line_no+=1
        if line_no==1:
            continue
        parts=line.strip().split('\t')
        if parts[11]=='TRAIN':
            train_no+=1
            if parts[3]=='NEUTRAL':
                train_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'0'+'\t'+parts[4]+'\n')
            elif parts[3]=='ENTAILMENT':
                train_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'1'+'\t'+parts[4]+'\n')
            elif parts[3]=='CONTRADICTION':
                train_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'2'+'\t'+parts[4]+'\n')
        elif parts[11]=='TEST':
            test_no+=1
            if parts[3]=='NEUTRAL':
                test_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'0'+'\t'+parts[4]+'\n')
            elif parts[3]=='ENTAILMENT':
                test_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'1'+'\t'+parts[4]+'\n')
            elif parts[3]=='CONTRADICTION':
                test_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'2'+'\t'+parts[4]+'\n')
        elif parts[11]=='TRIAL':
            dev_no+=1
            if parts[3]=='NEUTRAL':
                dev_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'0'+'\t'+parts[4]+'\n')
            elif parts[3]=='ENTAILMENT':
                dev_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'1'+'\t'+parts[4]+'\n')
            elif parts[3]=='CONTRADICTION':
                dev_file.write(parts[1].lower()+'\t'+parts[2].lower()+'\t'+'2'+'\t'+parts[4]+'\n')
        
        else: 
            print line
            remain_no+=1
    print train_no, test_no, dev_no, train_no+test_no+dev_no
    read_file.close()
    train_file.close()
    test_file.close()

def Extract_Vocab(path, train, dev, test):
    #consider all words, including unknown from word2vec, because some sentence 
    '''
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
    '''
    files=[train, dev, test]
    writeFile=open(path+'vocab.txt', 'w')
    vocab={}
    count=0
    max_length=0 # result 32
    for file in files:
        readFile=open(path+file, 'r')
        for line in readFile:
            tokens=line.strip().split('\t')
            for i in [0,1]:
                words=tokens[i].strip().split()
                if len(words)> max_length:
                    max_length=len(words)
                for word in words:
                    key=vocab.get(word)
                    if key is None:
                        count+=1
                        vocab[word]=count
                        writeFile.write(str(count)+'\t'+word+'\n')
                        
        readFile.close()
    writeFile.close()
    print 'total words: ', count
    print 'max_length: ', max_length

def transcate_word2vec_into_entailment_vocab(rootPath):
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
        sub_dict = [(prob, label) for prob, label in izip(sub_probs, sub_labels)] # a list of tuple
        #sorted_probs=sorted(sub_probs, reverse = True)
        sorted_tuples=sorted(sub_dict,key=lambda tup: tup[0], reverse = True) 
        map=0.0
        find=False
        corr_no=0
        #MAP
        for index, (prob,label) in enumerate(sorted_tuples):
            if label==1:
                corr_no+=1 # the no of correct answers
                all_corr_answer+=1
                map+=1.0*corr_no/(index+1)
                find=True
        #MRR
        for index, (prob,label) in enumerate(sorted_tuples):
            if label==1:
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
              
def reform_for_bleu_nist(trainFile):#not useful
    #first src file
    read_train=open(trainFile, 'r')
    write_src=open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/train_src.xml', 'w')
    write_src.write('<mteval>'+'\n'+'<srcset setid="WMT08" srclang="Czech">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_src.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[0])+' </seg>\n</p>\n')
        id+=1
    write_src.write('</doc>\n</srcset>\n</mteval>\n')
    write_src.close()
    read_train.close()
    #second, ref
    read_train=open(trainFile, 'r')
    write_ref=open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/train_ref.xml', 'w')
    write_ref.write('<mteval>'+'\n'+'<refset setid="WMT08" srclang="Czech" trglang="English" refid="reference01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_ref.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[0])+' </seg>\n</p>\n')
        id+=1
    write_ref.write('</doc>\n</refset>\n</mteval>\n')
    write_ref.close()
    read_train.close()     
    #third, sys
    read_train=open(trainFile, 'r')
    write_sys=open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/train_sys.xml', 'w')
    write_sys.write('<mteval>'+'\n'+'<tstset setid="WMT08" srclang="Czech" trglang="English" sysid="system01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_sys.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[1])+' </seg>\n</p>\n')
        id+=1
    write_sys.write('</doc>\n</tstset>\n</mteval>\n')
    write_sys.close()
    read_train.close()        
        
def putAllMtTogether():
    pathroot='/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST'
    train_files=[#pathroot+'/result_train/BLEU1-seg.scr',
                 #pathroot+'/result_train/BLEU2-seg.scr',pathroot+'/result_train/BLEU3-seg.scr',
                 pathroot+'/result_train/BLEU4-seg.scr',
                 #pathroot+'/result_train/NIST1-seg.scr',
                 #pathroot+'/result_train/NIST2-seg.scr',
                 #pathroot+'/result_train/NIST3-seg.scr',pathroot+'/result_train/NIST4-seg.scr',
                 pathroot+'/result_train/NIST5-seg.scr'
                 ]
    
    test_files=[#pathroot+'/result_test/BLEU1-seg.scr',
                #pathroot+'/result_test/BLEU2-seg.scr',pathroot+'/result_test/BLEU3-seg.scr',
                pathroot+'/result_test/BLEU4-seg.scr',
                 #pathroot+'/result_test/NIST1-seg.scr',
                 #pathroot+'/result_test/NIST2-seg.scr',
                 #pathroot+'/result_test/NIST3-seg.scr',pathroot+'/result_test/NIST4-seg.scr',
                 pathroot+'/result_test/NIST5-seg.scr',
                  #pathroot+'maxsim-v1.01/paraphrase/test.score'
                  ]

    #posi=[4, 4,4,4,4,  4,4,4,4]
    posi=[4, 4]
    size=len(posi)
    
    train_write=open(pathroot+'/result_train/concate_2mt_train.txt', 'w')
    scores=[]
    for i in range(size):
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
        for i in range(size):
            train_write.write(values_matrix[i,j]+'\t')
        train_write.write('\n')
    train_write.close()
    #test
    test_write=open(pathroot+'/result_test/concate_2mt_test.txt', 'w')
    scores=[]
    for i in range(size):
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
        for i in range(size):
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
            for i in [0,1]:
                sent2set=set(parts[i].split())# do not consider repetition
                for word in sent2set:
                    if word not in stop_words:
                        count=word2idf.get(word,0)
                        word2idf[word]=count+1
        read_file.close()
        
    '''   
    #train file
    read_train=open(path+trainfile, 'r')
    write_train=open(path+'train_word_matching_scores.txt','w')
    for line in read_train:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[0].split()
        answer=parts[1].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        write_train.write(str(WC)+' '+str(WWC)+'\n')
    write_train.close()
    read_train.close()
    
    #test file
    read_test=open(path+testfile, 'r')
    write_test=open(path+'test_word_matching_scores.txt','w')
    for line in read_test:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[0].split()
        answer=parts[1].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        write_test.write(str(WC)+' '+str(WWC)+'\n')
    write_test.close()
    read_test.close()             
    print 'two word matching values generated' 
    '''
    WC_train=[]
    WWC_train=[]
    #train file
    read_train=open(path+trainfile, 'r')
    #write_train=open(path+'train_word_matching_scores_normalized.txt','w')
    for line in read_train:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[0].split()
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
        question=parts[0].split()
        answer=parts[1].split()
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
    
    write_train=open(path+'train_word_matching_scores_normalized.txt','w')
    for index,wc in enumerate(WC_train):
        wc=(wc-min_WC)*1.0/(max_WC-min_WC)
        write_train.write(str(wc)+' '+str(WWC_train[index])+'\n')
    write_train.close()
    write_test=open(path+'test_word_matching_scores_normalized.txt','w')
    for index,wc in enumerate(WC_test):
        wc=(wc-min_WC)*1.0/(max_WC-min_WC)
        write_test.write(str(wc)+' '+str(WWC_test[index])+'\n')
    write_test.close()    
    
    print 'two word matching values generated' 
    
if __name__ == '__main__':
    path='/mounts/data/proj/wenpeng/Dataset/SICK/'
    #extract_pairs(path, 'SICK.txt')
    #Extract_Vocab(path, 'train.txt', 'dev.txt', 'test.txt')
    transcate_word2vec_into_entailment_vocab(path)
    #compute_map_mrr(path+'test_filtered.txt')
    #reform_for_bleu_nist(path+'WikiQA-train.txt')
    #putAllMtTogether()
    #two_word_matching_methods(path, 'WikiQA-train.txt', 'test_filtered.txt')
    #
    


