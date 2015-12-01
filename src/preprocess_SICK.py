import numpy
from itertools import izip
from xml.sax.saxutils import escape
#from nltk.tokenize import TreebankWordTokenizer
from sklearn import svm
from scipy import spatial
from numpy import linalg as LA
from nltk.corpus import wordnet as wn
import collections

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
    writeFile=open(path+'vocab_nonoverlap_train_plus_dev.txt', 'w')
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
    readFile=open(rootPath+'vocab_nonoverlap_train_plus_dev.txt', 'r')
    writeFile=open(rootPath+'vocab_nonoverlap_train_plus_dev_in_word2vec_embs_300d.txt', 'w')
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
              
def reform_for_bleu_nist(path, trainFile, file_title):#not useful
    #first src file
    read_train=open(path+trainFile, 'r')
    write_src=open(path+file_title+'_src.xml', 'w')
    write_src.write('<mteval>'+'\n'+'<srcset setid="WMT08" srclang="Czech">'+'\n'+'<doc docid="train" genre="nw">'+'\n')

    write_ref=open(path+file_title+'_ref.xml', 'w')
    write_ref.write('<mteval>'+'\n'+'<refset setid="WMT08" srclang="Czech" trglang="English" refid="reference01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')

    write_sys=open(path+file_title+'_sys.xml', 'w')
    write_sys.write('<mteval>'+'\n'+'<tstset setid="WMT08" srclang="Czech" trglang="English" sysid="system01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_src.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[0])+' </seg>\n</p>\n')
        write_ref.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[0])+' </seg>\n</p>\n')
        write_sys.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[1])+' </seg>\n</p>\n')
        id+=1
    write_src.write('</doc>\n</srcset>\n</mteval>\n')
    write_src.close()
    write_ref.write('</doc>\n</refset>\n</mteval>\n')
    write_ref.close()
    write_sys.write('</doc>\n</tstset>\n</mteval>\n')
    write_sys.close()
    read_train.close()
    '''
    #second, ref
    read_train=open(trainFile, 'r')
    write_ref=open(path+file_title+'_ref.xml', 'w')
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
    write_sys=open(path+file_title+'_sys.xml', 'w')
    write_sys.write('<mteval>'+'\n'+'<tstset setid="WMT08" srclang="Czech" trglang="English" sysid="system01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_sys.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[1])+' </seg>\n</p>\n')
        id+=1
    write_sys.write('</doc>\n</tstset>\n</mteval>\n')
    write_sys.close()
    read_train.close()       
    ''' 
def reform_for_maxsim(path, trainFile, file_title):#not useful
    #first src file
    read_train=open(path+trainFile, 'r')
    write_src=open(path+file_title+'_src.txt', 'w')
    #write_src.write('<mteval>'+'\n'+'<srcset setid="WMT08" srclang="Czech">'+'\n'+'<doc docid="train" genre="nw">'+'\n')

    write_ref=open(path+file_title+'_ref.txt', 'w')
    #write_ref.write('<mteval>'+'\n'+'<refset setid="WMT08" srclang="Czech" trglang="English" refid="reference01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')

    write_sys=open(path+file_title+'_sys.txt', 'w')
    #write_sys.write('<mteval>'+'\n'+'<tstset setid="WMT08" srclang="Czech" trglang="English" sysid="system01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    #id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_src.write(parts[0]+'\n')
        write_ref.write(parts[0]+'\n')
        write_sys.write(parts[1]+'\n')
        #id+=1
    #write_src.write('</doc>\n</srcset>\n</mteval>\n')
    write_src.close()
    #write_ref.write('</doc>\n</refset>\n</mteval>\n')
    write_ref.close()
    #write_sys.write('</doc>\n</tstset>\n</mteval>\n')
    write_sys.close()
    read_train.close()
    '''
    #second, ref
    read_train=open(trainFile, 'r')
    write_ref=open(path+file_title+'_ref.xml', 'w')
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
    write_sys=open(path+file_title+'_sys.xml', 'w')
    write_sys.write('<mteval>'+'\n'+'<tstset setid="WMT08" srclang="Czech" trglang="English" sysid="system01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_sys.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[1])+' </seg>\n</p>\n')
        id+=1
    write_sys.write('</doc>\n</tstset>\n</mteval>\n')
    write_sys.close()
    read_train.close()       
    '''         

def reform_for_terp(path, trainFile, file_title):#not useful
    #first src file
    read_train=open(path+trainFile, 'r')
    #write_src=open(path+file_title+'_src.sgm', 'w')
    #write_src.write('<mteval>'+'\n'+'<srcset setid="WMT08" srclang="Czech">'+'\n'+'<doc docid="train" genre="nw">'+'\n')

    write_ref=open(path+file_title+'_ref.sgm', 'w')
    write_ref.write('<refset setid="WMT08" srclang="Czech" trglang="English">'+'\n'+'<doc docid="train" genre="nw"  refid="reference01">'+'\n')

    write_sys=open(path+file_title+'_sys.sgm', 'w')
    write_sys.write('<tstset setid="WMT08" srclang="Czech" trglang="English">'+'\n'+'<doc docid="train" genre="nw"  sysid="system01">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        #write_src.write('<seg id="'+str(id)+'">'+' '+escape(parts[0])+' </seg>\n')
        write_ref.write('<seg id="'+str(id)+'">'+' '+escape(parts[0])+' </seg>\n')
        write_sys.write('<seg id="'+str(id)+'">'+' '+escape(parts[1])+' </seg>\n')
        id+=1
    #write_src.write('</doc>\n</srcset>\n</mteval>\n')
    #write_src.close()
    write_ref.write('</doc>\n</refset>\n')
    write_ref.close()
    write_sys.write('</doc>\n</tstset>\n')
    write_sys.close()
    read_train.close()
 
def putAllMtTogether():
    pathroot='/mounts/data/proj/wenpeng/Dataset/SICK/'
    train_files=[pathroot+'Train_plus_dev_MT/Badger-seg.scr', pathroot+'Train_plus_dev_MT/BLEU1-seg.scr',
                 pathroot+'Train_plus_dev_MT/BLEU2-seg.scr',pathroot+'Train_plus_dev_MT/BLEU3-seg.scr',
                 pathroot+'Train_plus_dev_MT/BLEU4-seg.scr',
                 pathroot+'Train_plus_dev_MT/NIST1-seg.scr',pathroot+'Train_plus_dev_MT/NIST2-seg.scr',
                 pathroot+'Train_plus_dev_MT/NIST3-seg.scr',pathroot+'Train_plus_dev_MT/NIST4-seg.scr',
                 pathroot+'Train_plus_dev_MT/NIST5-seg.scr',
                  pathroot+'Train_plus_dev_MT/maxsim.score',
                 pathroot+'Train_plus_dev_MT/meteor.txt', pathroot+'Train_plus_dev_MT/sepia-seg.scr',
                 pathroot+'Train_plus_dev_MT/terpa.seg.scr'
                 ]
    
    test_files=[pathroot+'Test_MT/Badger-seg.scr', pathroot+'Test_MT/BLEU1-seg.scr',
                 pathroot+'Test_MT/BLEU2-seg.scr',pathroot+'Test_MT/BLEU3-seg.scr',
                 pathroot+'Test_MT/BLEU4-seg.scr',
                 pathroot+'Test_MT/NIST1-seg.scr',pathroot+'Test_MT/NIST2-seg.scr',
                 pathroot+'Test_MT/NIST3-seg.scr',pathroot+'Test_MT/NIST4-seg.scr',
                 pathroot+'Test_MT/NIST5-seg.scr',
                  pathroot+'Test_MT/maxsim.score',
                 pathroot+'Test_MT/meteor.txt', pathroot+'Test_MT/sepia-seg.scr',
                 pathroot+'Test_MT/terpa.seg.scr'
                 ]

    posi=[4, 4,4,4,4,  4,4,4,4,4,  1,3,4,4]
    size=len(posi)
    
    train_write=open(pathroot+'/Train_plus_dev_MT/concate_14mt_train.txt', 'w')
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
    '''
    test_write=open(pathroot+'/Test_MT/concate_14mt_test.txt', 'w')
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
    '''
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
def test_mt_metrics(train_label, test_label):
    #first load mt metrics
    pathroot='/mounts/data/proj/wenpeng/Dataset/SICK/'
    train_files=[
                pathroot+'Train_MT/Badger-seg.scr', 
#                  pathroot+'Train_MT/BLEU1-seg.scr',
#                  pathroot+'Train_MT/BLEU2-seg.scr',
#                  pathroot+'Train_MT/BLEU3-seg.scr',
#                  pathroot+'Train_MT/BLEU4-seg.scr',
                pathroot+'Train_MT/NIST1-seg.scr',
                pathroot+'Train_MT/NIST2-seg.scr',
                pathroot+'Train_MT/NIST3-seg.scr',
                pathroot+'Train_MT/NIST4-seg.scr',
                pathroot+'Train_MT/NIST5-seg.scr',
                  pathroot+'Train_MT/maxsim.score',
                 pathroot+'Train_MT/meteor.txt', 
                 pathroot+'Train_MT/sepia-seg.scr'
                 ]
    
    test_files=[
                pathroot+'Test_MT/Badger-seg.scr', 
#                 pathroot+'Test_MT/BLEU1-seg.scr',
#                  pathroot+'Test_MT/BLEU2-seg.scr',
#                  pathroot+'Test_MT/BLEU3-seg.scr',
#                  pathroot+'Test_MT/BLEU4-seg.scr',
                pathroot+'Test_MT/NIST1-seg.scr',
                pathroot+'Test_MT/NIST2-seg.scr',
                pathroot+'Test_MT/NIST3-seg.scr',
                pathroot+'Test_MT/NIST4-seg.scr',
                pathroot+'Test_MT/NIST5-seg.scr',
                  pathroot+'Test_MT/maxsim.score',
                 pathroot+'Test_MT/meteor.txt', 
                 pathroot+'Test_MT/sepia-seg.scr'
                 ]

    posi=[
        4,
#            4,
#            4,
#            4,
#            4,  
            4,
            4,
            4,
            4,
            4,  
           1,
           3,
           4
           ]
    size=len(posi)
    
    #train_write=open(pathroot+'/Train_MT/concate_14mt_train.txt', 'w')
    scores=[]
    for i in range(size):
        read_file=open(train_files[i], 'r')
        list_values=[]
        for line in read_file:
            tokens=line.strip().split()
            list_values.append(tokens[posi[i]])
        read_file.close()
        scores.append(list_values)
    train_matrix=numpy.array(scores).T
    '''
    col=values_matrix.shape[1]
    for j in range(col):
        for i in range(size):
            train_write.write(values_matrix[i,j]+'\t')
        train_write.write('\n')
    train_write.close()
    '''
    #test
    #test_write=open(pathroot+'/Test_MT/concate_14mt_test.txt', 'w')
    scores=[]
    for i in range(size):
        read_file=open(test_files[i], 'r')
        list_values=[]
        for line in read_file:
            tokens=line.strip().split()
            list_values.append(tokens[posi[i]])
        read_file.close()
        scores.append(list_values)
    test_matrix=numpy.array(scores).T
    '''
    col=values_matrix.shape[1]
    for j in range(col):
        for i in range(size):
            test_write.write(values_matrix[i,j]+'\t')
        test_write.write('\n')
    test_write.close()
    '''
    print 'mts loaded over'           
    
    
    
    train_l=[]
    read_file=open(train_label, 'r')
    for line in read_file:
        train_l.append(int(line.strip().split('\t')[2]))
    read_file.close()
    '''
    read_file=open(train_values, 'r')
    train_matrix=[]
    for line in read_file:
        train_matrix.append(map(float,line.strip().split()))
    read_file.close()
    '''
    #test
    test_l=[]
    read_file=open(test_label, 'r')
    for line in read_file:
        test_l.append(int(line.strip().split('\t')[2]))
    read_file.close()
    '''
    read_file=open(test_values, 'r')
    test_matrix=[]
    for line in read_file:
        test_matrix.append(map(float,line.strip().split()))
    read_file.close()
    '''
    
    #svm
    clf = svm.SVC(kernel='linear')#OneVsRestClassifier(LinearSVC()) #linear 76.11%, poly 75.19, sigmoid 66.50, rbf 73.33
    clf.fit(train_matrix, train_l)
    results=clf.predict(test_matrix)
    corr_count=0
    test_size=len(test_l)
    for i in range(test_size):
        if results[i]==test_l[i]:
            corr_count+=1
    acc=corr_count*1.0/test_size
    print 'acc: ', acc, corr_count, '/', test_size    
    '''
    
    #logistic regression
    est=LinearRegression().fit(train_matrix, train_l)
    results=est.predict(test_matrix)
    corr_count=0
    test_size=len(test_l)
    for i in range(test_size):
        if numpy.absolute(results[i]-test_l[i])<0.5:
            corr_count+=1
    acc=corr_count*1.0/test_size
    print 'acc: ', acc, corr_count, '/', test_size   
    '''
    '''
    #DecisionTreeClassifier
    est=DecisionTreeClassifier().fit(train_matrix, train_l)
    results=est.predict(test_matrix)
    corr_count=0
    test_size=len(test_l)
    for i in range(test_size):
        if results[i]==test_l[i]:
            corr_count+=1
    acc=corr_count*1.0/test_size
    print 'acc: ', acc, corr_count, '/', test_size  
    '''
    '''
    #KNeighborsClassifier
    est=KNeighborsClassifier().fit(train_matrix, train_l)
    results=est.predict(test_matrix)
    corr_count=0
    test_size=len(test_l)
    for i in range(test_size):
        if results[i]==test_l[i]:
            corr_count+=1
    acc=corr_count*1.0/test_size
    print 'acc: ', acc, corr_count, '/', test_size      
    '''
    '''
    #GaussianNB
    est=GaussianNB().fit(train_matrix, train_l)
    results=est.predict(test_matrix)
    corr_count=0
    test_size=len(test_l)
    for i in range(test_size):
        if results[i]==test_l[i]:
            corr_count+=1
    acc=corr_count*1.0/test_size
    print 'acc: ', acc, corr_count, '/', test_size     
    '''    
def combine_train_trial(path, trainfile, devfile):
    
    readtrain=open(path+trainfile, 'r')
    readdev=open(path+devfile, 'r')
    writefile=open(path+'train_plus_dev.txt', 'w')
    for line in readtrain:
        writefile.write(line.strip()+'\n')
    readtrain.close()
    for line in readdev:
        writefile.write(line.strip()+'\n')
    readdev.close()
    writefile.close()
    
def remove_overlap_words(path, inputfile, title):
    readfile=open(path+inputfile, 'r')
    writefile=open(path+title+'_removed_overlap.txt', 'w')
    for line in readfile:
        parts=line.strip().split('\t')
        overlap=set(parts[0].split()).intersection(set(parts[1].split()))
        sent1=' '
        for i in parts[0].split():
            if i not in overlap:
                sent1+=i+' '
        sent2=' '
        for i in parts[1].split():
            if i not in overlap:
                sent2+=i+' '  
        writefile.write(sent1+'\t'+sent2+'\t'+parts[2]+'\n')
    readfile.close()
    writefile.close()      
                
def features_for_nonoverlap_pairs(path, inputfile, title):
    
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

    syn_set=set()
    hyper_set=set()    
    anto_set=set()
    read_syn=open(path+'synonyms.txt', 'r')
    for line in read_syn:
        syn_set.add((line.strip().split()[0], line.strip().split()[1]))
    read_syn.close()
    read_hyper=open(path+'hypernyms.txt', 'r')
    for line in read_hyper:
        hyper_set.add((line.strip().split()[0], line.strip().split()[1]))
    read_hyper.close()
    read_anto=open(path+'antonyms.txt', 'r')
    for line in read_anto:
        anto_set.add((line.strip().split()[0], line.strip().split()[1]))
    read_anto.close()
            
    readfile=open(path+inputfile, 'r')
    writefile=open(path+title+'_rule_features_cosine_eucli_negation_len1_len2_syn_hyper1_hyper2_anto(newsimi0.4).txt', 'w')
    for line in readfile:
        parts=line.split('\t')
        sent1_emb= []
        if len(parts[0].strip())>0:
            for word in parts[0].split():
                emb=word2vec.get(word)
                if emb is not None:
                    sent1_emb.append(emb)
        sent2_emb= []
        if len(parts[1].strip())>0:
            for word in parts[1].split():
                emb=word2vec.get(word)
                if emb is not None:
                    sent2_emb.append(emb)  
        simi=0.0 # default if one is empty
        eucli=0.0
        if len(sent1_emb) > 0 and len(sent2_emb) > 0:
            simi=1 - spatial.distance.cosine(numpy.sum(numpy.array(sent1_emb), axis=0), numpy.sum(numpy.array(sent2_emb), axis=0))
            eucli=1.0/(1.0+spatial.distance.euclidean(numpy.sum(numpy.array(sent1_emb), axis=0), numpy.sum(numpy.array(sent2_emb), axis=0)))
        elif len(sent1_emb) > 0:
            eucli=1.0/(1.0+LA.norm(numpy.sum(numpy.array(sent1_emb), axis=0)))
        elif len(sent2_emb) > 0:
            eucli=1.0/(1.0+LA.norm(numpy.sum(numpy.array(sent2_emb), axis=0)))
        
        negation=0.0
        sent_concate=parts[0].split()+parts[1].split()
        #if ('no' in sent_concate or 'not' in sent_concate or 'nobody' in sent_concate or "isn't" in sent_concate) and (len(parts[0].strip().split()) < 5 and  len(parts[1].strip().split()) < 5  ):
        if ('no' in sent_concate or 'not' in sent_concate or 'nobody' in sent_concate or "isn't" in sent_concate):
            negation=1.0
        digit=0.0
        if ('one' in sent_concate or 'two' in sent_concate or 'three' in sent_concate or 'four' in sent_concate or 'five' in sent_concate):
            digit=1.0
        #len1_w=len(parts[0].strip().split())
        #len2_w=len(parts[1].strip().split())
        len1_c=len(parts[0].strip())
        len2_c=len(parts[1].strip())
        #synonym and hypernym, antonyms

        num_syn=0.0
        num_hyp1=0.0
        num_hyp2=0.0
        num_ant=0.0
        sent1=parts[0].strip().split()
        sent2=parts[1].strip().split()
        if len(sent1)>0 and len(sent2)>0:
            for token1 in sent1:
                for token2 in sent2:
                    if (token1, token2) in syn_set or (token2, token1) in syn_set:
                        num_syn+=1
                    if (token1,token2) in hyper_set:
                        num_hyp1+=1
                    if (token2,token1) in hyper_set:
                        num_hyp2+=1
                    if (token1,token2) in anto_set or (token2,token1) in anto_set:
                        num_ant+=1
                
        writefile.write(str(simi)+'\t'+str(eucli)+'\t'+str(negation)+'\t'+str(len1_c)+'\t'+str(len2_c)+'\t'+str(num_syn)+'\t'+str(num_hyp1)+'\t'+str(num_hyp2)+'\t'+str(num_ant)+'\n')
    writefile.close()
    readfile.close()
                  
def analysis(vector):
    sum=numpy.sum(vector)
    normalized_v=numpy.array([1.0/3, 1.0/3, 1.0/3])
    if sum > 0.0:
        normalized_v=vector/sum
    std=numpy.std(normalized_v)
    return std, normalized_v
            
def discriminative_weights(path, trainfile, testfile):
    #load word embeddings
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
    #
    train_pairs=[]
    pair2co_list={}
    readtrain=open(path+trainfile, 'r')
    for line in readtrain:
        parts=line.split('\t')
        sent1=parts[0].strip().split()
        sent2=parts[1].strip().split()
        label=parts[2].strip()
        pair_set=set()
        if len(sent1)>0 and len(sent2)>0:
            for i in sent1:
                for j in sent2:
                    pair_set.add((i,j))
        elif len(sent1)==0 and len(sent2)>0:
            for j in sent2:
                pair_set.add(('<empty>',j))
        elif len(sent2)==0 and len(sent1)>0:
            for i in sent1:
                pair_set.add((i,'<empty>'))
        else: # both are empty
            pair_set.add(('<empty>','<empty>'))
        
        train_pairs.append(pair_set)
        for pair in pair_set:
            listt=pair2co_list.get(pair)
            if listt is None:
                listt=numpy.zeros(3)
            
            if label == '0':
                listt[0]+=1.0
            elif label=='1':
                listt[1]+=1.0
            else:
                listt[2]+=1.0
            pair2co_list[pair]=listt
        #print pair2co_list
        #exit(0)
    readtrain.close()
    #now, start to form train features
    writetrain=open(path+'train_plus_dev_discri_features_0.3.txt', 'w')
    for pair_set in train_pairs:
        #print pair_set
        #exit(0)
        pair_sum=[]
        for pair in pair_set:
            #print pair, pair2co_list.get(pair)
            discri_w, distribu=analysis(pair2co_list.get(pair))
            emb1=word2vec.get(pair[0])
            emb2=word2vec.get(pair[1])
            if emb1 is not None and emb2 is not None:
                discri_w=1 - spatial.distance.cosine(emb1, emb2)
                if discri_w < 0.3: # we only consider really related words
                    discri_w=0.0
            else:
                discri_w=1.0
            pair_sum.append(discri_w*distribu)
        #exit(0)
        pair_sum=numpy.sum(numpy.array(pair_sum), axis=0)
        writetrain.write(' '.join(map(str,pair_sum))+'\n')
        #writetrain.write(str(numpy.argmax(pair_sum))+'\n')
    writetrain.close()
    #now, start to form test features
    #test_pairs=[]
    #pair2co_list={}
    '''
    writetest=open(path+'test_discri_features_0.3.txt', 'w')
    readtest=open(path+testfile, 'r')
    for line in readtest:
        parts=line.split('\t')
        sent1=parts[0].strip().split()
        sent2=parts[1].strip().split()
        #label=parts[2].strip()
        pair_set=set()
        if len(sent1)>0 and len(sent2)>0:
            for i in sent1:
                for j in sent2:
                    pair_set.add((i,j))
        elif len(sent1)==0 and len(sent2)>0:
            for j in sent2:
                pair_set.add(('<empty>',j))
        elif len(sent2)==0 and len(sent1)>0:
            for i in sent1:
                pair_set.add((i,'<empty>'))
        else: # both are empty
            pair_set.add(('<empty>','<empty>'))
        
        #train_pairs.append(pair_set)
        pair_sum=[]
        for pair in pair_set:
            listt=pair2co_list.get(pair)
            if listt is None:
                continue
            else:
                discri_w, distribu=analysis(listt)
                emb1=word2vec.get(pair[0])
                emb2=word2vec.get(pair[1])
                if emb1 is not None and emb2 is not None:
                    discri_w=1 - spatial.distance.cosine(emb1, emb2)
                    if discri_w < 0.3: # we only consider really related words
                        discri_w=0.0
                else:
                    discri_w=1.0
                pair_sum.append(discri_w*distribu)
        if len(pair_sum)==0:#all pairs are unknown
            pair_sum.append(numpy.array([0.56, 0.28,0.16]))
        pair_sum=numpy.sum(numpy.array(pair_sum), axis=0)       
        writetest.write(' '.join(map(str,pair_sum))+'\n')    
        #writetest.write(str(numpy.argmax(pair_sum))+'\n')    
    
    readtest.close()
    writetest.close()
    '''
                    
def use_nonoverlap_dataset(path, trainfile, testfile):
    readtrain=open(path+trainfile, 'r')
    writetrain=open(path+'train_plus_dev_removed_overlap_as_training.txt', 'w')
    for line in readtrain:
        parts=line.split('\t')
        sent1=parts[0].strip()
        sent2=parts[1].strip()
        if len(sent1) ==0:
            sent1='<empty>'
        if len(sent2)==0:
            sent2='<empty>'
        writetrain.write(sent1+'\t'+sent2+'\t'+parts[2].strip()+'\n')
    writetrain.close()
    readtrain.close()    
    '''
    readtest=open(path+testfile, 'r')
    writetest=open(path+'test_removed_overlap_as_training.txt', 'w')
    for line in readtest:
        parts=line.split('\t')
        sent1=parts[0].strip()
        sent2=parts[1].strip()
        if len(sent1) ==0:
            sent1='<empty>'
        if len(sent2)==0:
            sent2='<empty>'
        writetest.write(sent1+'\t'+sent2+'\t'+parts[2].strip()+'\n')
    writetest.close()
    readtest.close()         
    '''    
            
    
def extract_synonyms_for_token(token):
    synonyms_dog=[]
    for synset in wn.synsets(token):
        synonyms_dog+=synset.lemma_names()            
    return set(synonyms_dog)        

def extract_hypernyms_for_token(token):
    results=[]
    synsetss=wn.synsets(token)
    if len(synsetss)>0:
        apple = synsetss[0]
        hyperapple = set([i for i in apple.closure(lambda s:s.hypernyms())])
        for word in hyperapple:
            results+=word.lemma_names()
    return results                   

def syn_relation(token1, token2):
    syn1=extract_synonyms_for_token(token1)
    syn2=extract_synonyms_for_token(token2)
    #print syn1
    #print syn2
    if token1 in syn2 or token2 in syn1:
        return True
    else:
        return False
def hyper_relation(token1, token2):
    hyper1=extract_hypernyms_for_token(token1)
    if token2 in hyper1:
        return True
    else:
        return False

def extract_synonyms_hypernyms_antonyms(path, trainfile, testfile):
    readtrain=open(path+trainfile, 'r')
    readtest=open(path+testfile, 'r')
    writesyn=open(path+'synonyms.txt', 'w')
    writehyper=open(path+'hypernyms.txt', 'w')
    writeanto=open(path+'antonyms.txt', 'w')
    pair_train=set()
    pair_train_entail=set()
    pair_train_nonentail=set()
    pair_train_freq=collections.defaultdict(int)
    pair_test=set()
    syn_set=set()
    hyper_set=set()
    for line in readtrain:
        parts=line.split('\t')
        sent1=parts[0].strip().split()
        sent2=parts[1].strip().split()
        if len(sent1) >0 and len(sent2) >0:
            for token1 in sent1:
                for token2 in sent2:
                    pair_train.add((token1, token2))
                    if parts[2]=='1':
                        pair_train_entail.add((token1, token2))
                    else:
                        pair_train_nonentail.add((token1, token2))
                    pair_train_freq[(token1,token2)]+=1
    readtrain.close()
    for line in readtest:
        parts=line.split('\t')
        sent1=parts[0].strip().split()
        sent2=parts[1].strip().split()
        if len(sent1) >0 and len(sent2) >0:
            for token1 in sent1:
                for token2 in sent2:
                    pair_test.add((token1, token2))
    readtest.close()
    #filter
    both_set=pair_train | pair_test
    for (token1,token2) in both_set:
        if syn_relation(token1, token2):
            syn_set.add((token1, token2))
            writesyn.write(token1+'\t'+token2+'\n')
        if hyper_relation(token1, token2):
            hyper_set.add((token1, token2))
            writehyper.write(token1+'\t'+token2+'\n')
    writesyn.close()
    writehyper.close()
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
    possible_anto=pair_train_nonentail-pair_train_entail-syn_set-hyper_set
    for (token1,token2) in possible_anto:
        if pair_train_freq.get((token1,token2))>=2:
            emb1=word2vec.get(token1)
            emb2=word2vec.get(token2)
            if emb1 is not None and emb2 is not None:
                simi=1 - spatial.distance.cosine(emb1, emb2)
                if simi>0.4:
                    writeanto.write(token1+'\t'+token2+'\n')
    writeanto.close()
    print 'over'
        
    
    
    
if __name__ == '__main__':
    path='/mounts/data/proj/wenpeng/Dataset/SICK/'
    #extract_pairs(path, 'SICK.txt')
    #Extract_Vocab(path, 'train.txt', 'dev.txt', 'test.txt')
    #transcate_word2vec_into_entailment_vocab(path)
    #compute_map_mrr(path+'test_filtered.txt')
    #reform_for_bleu_nist(path, 'train_plus_dev.txt', 'train_plus_dev')
    #reform_for_maxsim(path, 'train_plus_dev.txt', 'train_plus_dev')
    #reform_for_terp(path, 'train_plus_dev.txt', 'train_plus_dev')
    #putAllMtTogether()
    #two_word_matching_methods(path, 'WikiQA-train.txt', 'test_filtered.txt')
    #test_mt_metrics(path+'train.txt',  path+'test.txt') # found terp is not helpful
    #combine_train_trial(path, 'train.txt', 'dev.txt')
    #remove_overlap_words(path, 'train_plus_dev.txt', 'train_plus_dev')
    features_for_nonoverlap_pairs(path, 'test_removed_overlap.txt', 'test')
    #discriminative_weights(path, 'train_plus_dev_removed_overlap.txt', 'test_removed_overlap.txt')
    #use_nonoverlap_dataset(path, 'train_plus_dev_removed_overlap.txt', 'test_removed_overlap.txt') #maxlength 24, maxlentsh for train plus dev is also 24
    #Extract_Vocab(path, 'train_plus_dev_removed_overlap_as_training.txt', 'train_plus_dev_removed_overlap_as_training.txt', 'test_removed_overlap_as_training.txt')
    #transcate_word2vec_into_entailment_vocab(path)
    #extract_synonyms_hypernyms_antonyms(path, 'train_plus_dev_removed_overlap_as_training.txt', 'test_removed_overlap_as_training.txt')
    


