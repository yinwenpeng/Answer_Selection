import random

def recoverTxT():
    path='/mounts/data/proj/wenpeng/Dataset/insuranceQA/'
    ind2word={}
    readFile=open(path+'vocabulary', 'r')
    for line in readFile:
        tokens=line.strip().split()
        ind2word[tokens[0]]=tokens[1]
    readFile.close()
    print 'vocab size is: ', len(ind2word)
    #recover answers
    readFile=open(path+'answers.label.token_idx', 'r')
    writeFile=open(path+'ywp_index2answer.txt', 'w')
    for line in readFile:
        tokens=line.strip().split()
        length=len(tokens)
        writeFile.write(tokens[0]+'\t')
        for ind in tokens[1:]:
            writeFile.write(ind2word.get(ind)+' ')
        writeFile.write('\n')
    writeFile.close()
    readFile.close()
    print 'answers recoverd'
    #recover questions
    readFile=open(path+'question.train.token_idx.label', 'r')
    writeFile=open(path+'ywp_question.train.answerindex.txt', 'w')
    for line in readFile:
        tokens=line.strip().split('\t')
        inds=tokens[0].strip().split()
        #answers=tokens[1].strip().split()
        for ind in inds:
            writeFile.write(ind2word.get(ind)+' ')
        writeFile.write('\t')
        writeFile.write(tokens[1]+'\n')
    writeFile.close()
    readFile.close()
    print 'question recoverd'
    #recover dev, test1, test2
    files=['question.dev.label.token_idx.pool', 'question.test1.label.token_idx.pool', 'question.test2.label.token_idx.pool']
    for file in files:
        readFile=open(path+file, 'r')
        writeFile=open(path+'ywp_'+file,'w')
        for line in readFile:
            parts=line.strip().split('\t')
            writeFile.write(parts[0]+'\t')
            inds=parts[1].strip().split()
            for ind in inds:
                writeFile.write(ind2word.get(ind)+' ')
            writeFile.write('\t'+parts[2]+'\n')
        writeFile.close()
        readFile.close()
    print 'all dev, test1 and test2 recoverd'
    

def reformat():
    #make ibm data into (label, que, answer)
    path='/mounts/data/proj/wenpeng/Dataset/insuranceQA/'
    readFile=open(path+'ywp_index2answer.txt', 'r')
    id2answer={}
    for line in readFile:
        tokens=line.strip().split('\t')
        id2answer[tokens[0]]=tokens[1].strip()
    readFile.close()
    print 'id2answers loaded over'
    files=['ywp_question.dev.label.token_idx.pool', 'ywp_question.test1.label.token_idx.pool', 'ywp_question.test2.label.token_idx.pool']
    for i, file in enumerate(files):
        print file, '......'
        readFile=open(path+file, 'r')
        if i==0:
            writeFile=open(path+'dev.txt', 'w')
        elif i==1:
            writeFile=open(path+'test1.txt', 'w')
        elif i==2:
            writeFile=open(path+'test2.txt', 'w')
        for line in readFile:
            tokens=line.strip().split('\t')
            gold_inds=tokens[0].strip().split()
            inds=tokens[2].strip().split()
            for id in inds:
                if id in gold_inds:
                    writeFile.write('1\t'+tokens[1].strip()+'\t'+id2answer.get(id)+'\n')
                else:
                    writeFile.write('0\t'+tokens[1].strip()+'\t'+id2answer.get(id)+'\n')
        writeFile.close()
        readFile.close()
    print 'dev, test1 and test2 reformat over'
    #reform train file
    readFile=open(path+'ywp_question.train.answerindex.txt', 'r')
    writeFile=open(path+'train.txt', 'w')
    for line in readFile:
        tokens=line.strip().split('\t')
        question=tokens[0]
        gold_inds=tokens[1].strip().split()
        gold_inds_set=set(gold_inds)
        samples=random.sample(id2answer.items(), 500)
        for gold_ind in gold_inds:
            for (ind, nega_answer) in samples:
                writeFile.write('1\t'+question+'\t'+id2answer.get(gold_ind)+'\n')
                if ind not in gold_inds_set:
                    #valid nega
                    writeFile.write('0\t'+question+'\t'+nega_answer+'\n')
                else:
                    continue
    writeFile.close()
    readFile.close()
    print 'train reformat over'
                     
    
        
        
        
        
if __name__ == '__main__':
    reformat()
            
    
    
    
    
    
    
    
    
    
    
    
        