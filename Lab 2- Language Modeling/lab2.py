import nltk
from nltk.corpus import brown
# import random  
import random 
import collections

def get_uniCount(tokens):
	unigram_table={}

	for token in tokens:
		if token in unigram_table:
			unigram_table[token]+=1
		else:
			unigram_table[token] =1

	return unigram_table, len(tokens)

def randomSentUniGram(length,tokens):
	ordered_tokens = list(set(sorted(tokens)))
	return  ' '.join(random.choice(ordered_tokens)  for i in range(length))
	

	uingram_table,num_tokens = get_uniCount(tokens)
	#need to insert probability of each word
	ordered_unigram_table = collections.OrderedDict(sorted(uingram_table.items()))
	
	probability_of_each_word = []
	for key,v in ordered_unigram_table.items():
		probability_of_each_word.append(v/num_tokens)

	print(random.choices(ordered_tokens,probability_of_each_word))
	return ' '.join(random.choices(ordered_tokens,probability_of_each_word)[0] for i in range(length))

def generateSentiGram(length,tokens):

	bigram_table,num_bigrams = get_biCounts(tokens)

	# pick a random word from the corpus to start with
	word = random.choice(list(bigram_table))
	# generate 15 more words
	sentLst =[]
	for i in range(length):
	    #print (word)
	    sentLst.append(word)
	    if word in bigram_table:
	    	#print("cfd[word].keys()",cfd[word].keys())
	        word = random.choice(list(bigram_table[word].keys()))
	
	return ' '.join(sentLst)

# given a genre, return a "dictionary with dictionaries inside" and the # of seen bigrams
# where outside keys are the first words in existing bigrams,
# and values are dictionaries with the subsequent word as key
# and counts of such bigram as value
def get_biCounts(tokens):
    uniCounts, length = get_uniCount(tokens)
    bigram_table = {}
    num_bigrams = 0
    for x in range(0, length - 1):
        if tokens[x] in bigram_table:
            if tokens[x + 1] in bigram_table[tokens[x]]:
                bigram_table[tokens[x]][tokens[x + 1]] += 1
            else:
                bigram_table[tokens[x]][tokens[x + 1]] = 1
                num_bigrams += 1
        else:
            bigram_table[tokens[x]] = {}
            bigram_table[tokens[x]][tokens[x + 1]] = 1
            num_bigrams += 1
    return bigram_table, num_bigrams
def getWordProbability(words,uniCounts, num_tokens):
	#get uniGram
	sents_probability =1
	for w in words:
		if w in uniCounts:
			sents_probability *= uniCounts[w]/num_tokens

	return sents_probability
	 

def perplexity(words,tokens):
	uniCounts, length = get_uniCount(tokens)
	prob = getWordProbability(words,uniCounts, length)
	return pow(prob, -1 / len(words))
# command line interpreter
if __name__=='__main__':

	ans= True
	# str_news_cont = nltk.Text(brown.words(categories=['news']))
	# print(str_news_cont)
	# str_hobbies_cont = nltk.Text(brown.words(categories=['hobbies']))
	news_cont = brown.sents(categories=['news'])
	hobbies_cont= brown.sents(categories=['hobbies'])
	sents_news_cont = [' '.join(sent) for sent in news_cont]
	str_news_cont =  ''.join([str(elem) for elem in sents_news_cont]) #[''.join(sent) for sent in sents_news_cont]

	sents_hobbies_cont = [' '.join(sent) for sent in hobbies_cont]
	str_hobbies_cont =  ''.join([str(elem) for elem in sents_hobbies_cont]) 

	news_tokens = nltk.word_tokenize(str_news_cont)
	hobbies_tokens = nltk.word_tokenize(str_hobbies_cont)
	while ans:
		print("""
	    1.Compute Unigram (news)
	    2.Compute Unigram (hobbies)
	    3.Compute bigram (news)
	    4-Compute bigram (hobbies)
	    5.genrate sentence using Unigram
	    6.genrate sentence using Bigram
	    7- calculate perplexity
	    8.Exit/Quit
	    """)
		ans=input("What would you like to do? ")
	
		if ans=="1":
			print("Unigram of News: ",get_uniCount(news_tokens))
		elif ans=="2":
			print("Unigram of News: ",get_uniCount(hobbies_tokens))
		elif ans=="3":
			print("Unigram of News: ",get_biCounts(news_tokens))
		elif ans=="4":
			print("Unigram of News: ",get_uniCount(hobbies_tokens))
		elif ans=="5":
			print("Sentence generation using Unigram:",randomSentUniGram(15,news_tokens))
		elif ans=="6":
			print("Sentence generation using Bigram:",generateSentiGram(15,news_tokens))
		elif ans=="7":
			print("perplexity(self, words,tokens)",perplexity(['i','want','you','to'],news_tokens))
		elif ans=="8":
			ans=False
		else:
			print("Please select one of availale option")


	#print(str_news_cont)
	#print("brown.words(categories=['news'])",brown.words(categories=['news']))
	
	#hobbies_tokens = nltk.word_tokenize(hobbies_cont)

	#
	#print(get_biCounts(news_tokens))
	#
	#print(generateSentiGram(15,news_tokens))


	