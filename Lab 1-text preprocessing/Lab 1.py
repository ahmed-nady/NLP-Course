from urllib.request import urlopen
import nltk
from bs4 import BeautifulSoup
import re
# Python program to find the k most frequent words 
# from data set 
from collections import Counter 

#Lab1: Text Preprocessing

article_url = "http://en.wikipedia.org/w/index.php?title=Natural+language+processing&action=raw"
article_file = "nlp.txt"
#2-Create a function that takes the name of the article and returns the raw text:
def download_wikipeda_article(article_url):

	the_raw_wiki_text = urlopen(article_url).read()
	return the_raw_wiki_text

def read_wikipeda_article(article_file):
	with open(article_file) as f:
		return f.read()

#Write a function that cleans the raw Wikipedia text, 
#before you can tokenize the text, you need to clean it from layout commands, formatting directives and other things that do not belong there:
def clean_wikipedia_article(article_cont):
	# cleaned_text = nltk.clean_html(wiki_text)
	# return cleaned_text
	return BeautifulSoup(article_cont,'html.parser').get_text()

def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned =re.sub(r"<ref (name=)? .* > .*(</ref>)"," ",cleaned)
    #contents = re.sub(r'\[\[File:[^\]]*? ', '', contents)
    return cleaned.strip()

def loading_corpus():
	from nltk.corpus import PlaintextCorpusReader
	corpus_root = "wiki_corpus"
	wordlists = PlaintextCorpusReader(corpus_root, '.*')
	print (wordlists.fileids())
	print( wordlists.words())
	tokens = nltk.word_tokenize(wordlists.raw() )
	print(tokens)

def tokenize(raw):

	tokens = re.split(r'[ \t\n]+',raw)
	return tokens

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 

#Finally, your program should print some interesting statistics for the given Wikipedia article
def print_statistics(tokenized_text):
	#How many words and wordforms are there?
	print("number of words: ",len(tokenized_text))
	#Average word length, longest words?
	#we get number of characters in each word
	words_len = [len(w) for w in tokenized_text]
	number_of_chars = sum(words_len)
	print("Average word length ",number_of_chars/len(tokenized_text))
	longest_word_index = words_len.index(max(words_len))
	longest_word = tokenized_text[longest_word_index]
	print("longest_word: ",longest_word)

	print("most_frequent",most_frequent(tokenized_text))
	 
	#10 most frequent words? How many percent?
	# Pass the split_it list to instance of Counter class. 
	counter = Counter(tokenized_text) 
	  
	# most_common() produces k frequently encountered 
	# input values and their respective counts. 
	most_occur = counter.most_common(10) 

	print("10 most frequent words: ",most_occur)

# A Dynamic Programming based Python program for edit 
# distance problem 
def editDistDP(str1, str2, m, n): 
	# Create a table to store results of subproblems 
	dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 

	# Fill d[][] in bottom up manner 
	for i in range(m + 1): 
		for j in range(n + 1): 

			# If first string is empty, only option is to 
			# insert all characters of second string 
			if i == 0: 
				dp[i][j] = j # Min. operations = j 

			# If second string is empty, only option is to 
			# remove all characters of second string 
			elif j == 0: 
				dp[i][j] = i # Min. operations = i 

			# If last characters are same, ignore last char 
			# and recur for remaining string 
			elif str1[i-1] == str2[j-1]: 
				dp[i][j] = dp[i-1][j-1] 

			# If last character are different, consider all 
			# possibilities and find minimum 
			else: 
				dp[i][j] = 1 + min(dp[i][j-1],	 # Insert 
								dp[i-1][j],	 # Remove 
								dp[i-1][j-1]) # Replace 

	return dp[m][n] 



# command line interpreter
if __name__=='__main__':

	#print(download_wikipeda_article(article_url))
	#print(read_wikipeda_article(article_file))
	#loading_corpus()
	#exit(1)
	# print(clean_wikipedia_article(read_wikipeda_article(article_file)))
	# exit(1)
	# raw_txt = read_wikipeda_article(article_file)
	# clean_text = clean_html(raw_txt)
	# tokens = tokenize(clean_text)
	# print_statistics(tokens)

	# Driver program 
	str1 = "sunday"
	str2 = "saturday"

	print("editDistDP",editDistDP(str1, str2, len(str1), len(str2))) 
 
	# # Write-Overwrites 
	# file1 = open("cleaned_vlp.txt","w")#write mode 
	# file1.write(clean_text) 
	# file1.close() 