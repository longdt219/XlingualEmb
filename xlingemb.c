// This is the extension of Word2Vec to work with multiple languages using dictionary
// For question contact: Long Duong (longdt219@gmail.com)


//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING], output_file_neg[MAX_STRING], dict_file[MAX_STRING], refemb_file[MAX_STRING];

char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
int **dict_hash_en, **dict_hash_it;
int *dict_size_en, *dict_size_it;
int *dict_allocated;

long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
int nosel = 0;
int seldist = 0;
int selall = 1; // Default select all context and middle word
int join = 1, join2 = 0, balance = 0, combine=0;
int selcnt = 0;
float reg_sen = 0;
int relcnt = 0;

real chance=1, alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
	vocab_max_size += 1000;
	vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }

  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
//  printf("Add word %s to vocab\n",word);

  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));

  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  // Reset vocab hash
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
      vocab[a].word = NULL;
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  // This reallocation causing errors ???
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  vocab_max_size = vocab_size; // LD : We have to set the vocab_max_size here the same as vocab_size
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  printf(" Reduce Vocab Size \n");
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else {
	  free(vocab[a].word);
	  vocab[a].word = NULL;
  }
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }

    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
    printf("Vocab size: %lld\n", vocab_size);
  if (debug_mode > 0) {
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

int split2 (const char *str, char* c, char **arr, int count){
	// count is the maximum number of match
	char* token;
	int i = 0;
	while ((token = strsep(&str, c)) != NULL)
	  {
		if (i >= count) {
			printf(" Split the string with more than %d appearance", count);
			break;//exit(1);
		}
		arr[i] =  malloc( sizeof(char) * (strlen(token) + 1));
		strcpy(arr[i], token);
		i++;
	  }
	return i;
}

void ModelInitialization(){
	// Read the embedding file and initialize the weights
	printf ("Initialize the word embeddings from %s\n", refemb_file);

	FILE *fin = fopen(refemb_file,"rb");
	char buf[4096];
	int found =0 ;
	while (fgets(buf,sizeof(buf),fin) != NULL){
		char **tokens =  malloc(sizeof(char*) * 2);
		//printf("%s\n",buf);
		// Trim the input (remove newline and space
		while (1){
			size_t len_buf = strlen(buf);
			if ((len_buf>0) && ((buf[len_buf-1] == '\n')|| (buf[len_buf-1] == ' ')))
				buf[len_buf -1] = '\0';
			else break;
		}
		// Split
		int c = split2(buf," ",&tokens, 2);
		if (c==2) {
			// The first row
			if (atoi(tokens[1]) != layer1_size){
				printf("The reference embeddings have different dimension %d vs %d\n", atoi(tokens[1]),layer1_size);
				exit(1);
			}
		}else{
			if (c != layer1_size +1) {
				printf(" C = %d\n",c);
				printf("Problem with reference embeddings with line %s \n", buf);
				exit(1);
			}
			int curent_word = SearchVocab(tokens[0]);
			if (curent_word != -1){
				int i;
				found ++;
				for (i=0; i<layer1_size; i++){
					syn0[curent_word * layer1_size + i] = atof(tokens[i+1]);
				}
			}
		}

	}
	fclose(fin);
	printf ("Found %d/%d = %f  \n",found, vocab_size, found / (1.0 * vocab_size));
}
char* concat(char *s1, char *d, char *s2)
{
    char *result = malloc(strlen(s1)+strlen(d) + strlen(s2)+1);//+1 for the zero-terminator
    strcpy(result, s1);
    strcat(result, d);
    strcat(result, s2);
    return result;
}

void ReadDict(){
	// Allocate the memory according to the vocab_max_size
	dict_hash_en = malloc(vocab_max_size * sizeof(int*));
	dict_hash_it = malloc(vocab_max_size * sizeof(int*));
	dict_size_en = calloc(vocab_max_size , sizeof(int));
	dict_size_it = calloc(vocab_max_size , sizeof(int));

	dict_allocated = calloc(vocab_max_size , sizeof(int));
	int DEFAULT_DICT_ALLOCATED = 10 ;
	int total_vocab = 0;
	int en_vocab =0;
	int it_vocab = 0;
	int count_pair = 0;
	FILE *fin = fopen(dict_file,"rb");
	char buf[1000];
	int initial_vocab_size = vocab_size;
	while (fgets(buf,1000,fin) != NULL){
		char **tokens  = malloc(sizeof(char*)*2);
		// Trim the input
		while (1){
			size_t len_buf = strlen(buf);
			if ((len_buf>0) && ((buf[len_buf-1] == '\n')|| (buf[len_buf-1] == ' ')))
				buf[len_buf -1] = '\0';
			else break;
		}

		int c = split2(buf,"\t",tokens, 2);
		if (c!=2) {
			printf("Something wrong with the dictionary");
			exit(1);
		}

		//printf("'%s' and '%s' \n",tokens[0],tokens[1]);
		int en_id = SearchVocab(tokens[0]);
		int it_id = SearchVocab(tokens[1]);
		//printf(" en %d vs it %d\n", en_id, it_id);
		if (en_id != -1) en_vocab ++;
		if (it_id != -1) it_vocab ++;

		if ((en_id != -1) && (it_id != -1)){
			count_pair ++;
			if (join2){
				// Add this to the vocab
	    		char* s_word = tokens[0];
	    		char* t_word = tokens[1];
	        	char delimiter[] = ":";
	    		char* sense_word = concat(s_word,delimiter,t_word);
	    		// Add to dictionary
	    		// printf("Adding to dict : %s\n",sense_word);
	    		int i = SearchVocab(sense_word);
	        	if (i == -1) {
	        		int a = AddWordToVocab(sense_word);
	        	    vocab[a].cn = (vocab[en_id].cn + vocab[it_id].cn)/2; // Avoid this be removed
	        	    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
	        	    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	        	}else vocab[i].cn ++;
	        	// Reduce Vocab size if necessory
	        	if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
			}

			// Allocate the size if not
			if (dict_size_en[en_id] == 0){
				// First Allocate with default size
				dict_hash_en[en_id] = calloc(DEFAULT_DICT_ALLOCATED , sizeof(int));
				dict_allocated[en_id] = DEFAULT_DICT_ALLOCATED;
				total_vocab ++;
			}
			if (dict_size_it[it_id] ==0){
				dict_hash_it[it_id] = calloc(DEFAULT_DICT_ALLOCATED , sizeof(int));
				dict_allocated[it_id] = DEFAULT_DICT_ALLOCATED;
				total_vocab ++;
			}
			// Check if we have enough memory, if not reallocate
			if (dict_allocated[en_id]<= dict_size_en[en_id] + 2){
				dict_hash_en[en_id] = realloc(dict_hash_en[en_id], (dict_allocated[en_id] + DEFAULT_DICT_ALLOCATED) * sizeof(int));
				dict_allocated[en_id] = dict_allocated[en_id] + DEFAULT_DICT_ALLOCATED;
			}
			if (dict_allocated[it_id]<= dict_size_it[it_id] + 2){
				dict_hash_it[it_id] = realloc(dict_hash_it[it_id],(dict_allocated[it_id] + DEFAULT_DICT_ALLOCATED) * sizeof(int));
				dict_allocated[it_id] = dict_allocated[it_id] + DEFAULT_DICT_ALLOCATED;
			}
			// Add to the list ?
			dict_hash_en[en_id][dict_size_en[en_id]] = it_id;
			dict_size_en[en_id] ++;
			dict_hash_it[it_id][dict_size_it[it_id]] = en_id;
			dict_size_it[it_id] ++;
		}
		else{
			// For testing only
//			printf("In dict but not in corpus : '%s' and '%s' with id: %d and %d \n",tokens[0],tokens[1], en_id, it_id);
		}
	}
	fclose(fin);
	// Check some hash function
	printf("Join vocabulary coverage: (%d/%d) = %f  with %d pairs\n",total_vocab, initial_vocab_size, total_vocab / (1.0 * initial_vocab_size), count_pair);

}


void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  // LD: this sort vocab and filter infrequent words
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
    
    if (output_file_neg[0] != 0){
    // Initialize it randomly
        for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            syn1neg[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
        }
    }
  // LD: this creat Huff tree only if we are using HS
  if (hs)
    CreateBinaryTree();
}

real calculate_similarity_context(real* context, int word2){
	float total_1=0, total_2=0, total=0;
	int i;
	for (i = 0; i < layer1_size; i++){
		real w1 = context[i];
		real w2 = syn0[word2 * layer1_size + i];
		total_1 += w1*w1;
		total_2 += w2*w2;
		total += w1 * w2;
	}
	return total / (sqrt(total_1) * sqrt(total_2));
}

real calculate_similarity(int word1, int word2){
	float total_1=0, total_2=0, total=0;
	int i;
	for (i = 0; i < layer1_size; i++){
		real w1 = syn0[word1 * layer1_size + i];
		real w2 = syn0[word2 * layer1_size + i];
		total_1 += w1*w1;
		total_2 += w2*w2;
		total += w1 * w2;
	}
	return total / (sqrt(total_1) * sqrt(total_2));
}

int get_translation(unsigned long long *next_random, int word,
		int size_candidate, int* candidates, real* context){

	//printf("Word %s has %d candidates \n",vocab[word].word,size_candidate);
	// Representation of word
	if (nosel){
		// Don't select it, just randomly choose
		*next_random = *next_random * (unsigned long long)25214903917 + 11;
		return candidates[*next_random % size_candidate];
	}
	else {
		// Select the candidate base on distribution
		int i;
		if (seldist == 1){
			float scores[size_candidate];
			float total = 0;
			for (i=0; i<size_candidate; i++){
				if (selcnt)
					scores[i] = calculate_similarity_context(context,candidates[i]);
				else
					scores[i] = calculate_similarity(word,candidates[i]);
				total += scores[i];
			}
			*next_random = (*next_random % 1000) * (unsigned long long)25214903917 + 11;
			float random_value = (*next_random % 1000) / 1000.0;
			for (i=0; i<size_candidate; i++){
				random_value = random_value - scores[i] / total;
				if (random_value <=0){
					return candidates[i];
					break;
				}
			}
		}
		else {
			float max_sim = -1;
			int max_candidate = -1;
    		for (i=0; i<size_candidate; i++){
    			real similarity = 0;
    			if ((selall)||(selcnt))
    				similarity = calculate_similarity_context(context, candidates[i]);
    			else
    				similarity = calculate_similarity(word,candidates[i]);

    			if (similarity > max_sim){
    				max_sim = similarity;
    				max_candidate = candidates[i];
    			}
    		}
    		//printf("Select translation %s for word %s \n",vocab[max_candidate].word,vocab[word].word);
    		return max_candidate;
		}
	}
	return word;
}

int replace_word_with_chance(unsigned long long *next_random, int word, real* context){
	// Get the next random
	*next_random = *next_random  * (unsigned long long)25214903917 + 11;
	if ((*next_random % 1000) / 1000.0 <= chance){
    	int* candidates;
    	int size_candidate = 0;
    	if (dict_size_en[word] > 0) {
    		candidates = dict_hash_en[word];
    		size_candidate = dict_size_en[word];
    	}
    	else
    		if (dict_size_it[word] >0){
    			candidates = dict_hash_it[word];
    			size_candidate = dict_size_it[word];
    		}
    	if (size_candidate > 0){
    		// LD: overwrite word
    		return get_translation(&next_random, word,size_candidate, candidates, context);
    	}
    }
    return word;
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  // Holding temporary values
  real *cntvec = (real *)calloc(layer1_size, sizeof(real));
  real *temp_h = (real *)calloc(layer1_size, sizeof(real));

  // Hold the current list of context
  int *context_vec = (int *)calloc(2 * window +5, sizeof(real));
  int context_size = 0;

  // Open file and seek to the place where this thread should work on
  FILE *fi = fopen(train_file, "rb");
  // LD: The big files are divided equally to n threads.
  // LD: note each threads will the the whole m - iter.
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  int replace = 0;
  int total_words = 0;

  while (1) {
	  // Update the progress for each 10000 words
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      // LD: Here reduce the learning rate with some scheme
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    // Read from the file to form a sentence with MAX_SENTENCE_LENGTH id words
    if (sentence_length == 0) {
      //printf("\n ---------------------- \n");
      while (1) {
        word = ReadWordIndex(fi);

        if (feof(fi)) break;
        if (word == -1) continue;
        //printf("%s ",vocab[word].word);
        word_count++;
        if (word == 0) break; // LD: break if reach end of line
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        // LD: Also replace the context with the translation
        next_random = next_random * (unsigned long long)25214903917 + 11;

        //printf("%s ",  vocab[word].word);
        int temp = word;
        if (relcnt) word = replace_word_with_chance(&next_random,word,NULL);
        //printf("%s ", vocab[word].word);
        if (temp!=word) replace ++;
        total_words  ++;

        sen[sentence_length] = word;
        sentence_length++;

        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }

      sentence_position = 0;
    }

    // Check condition for stop training, seek for next chunk of text for process
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break; // When we must go over iter then we stop
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    // Start training with this word
    word = sen[sentence_position];
    if (word == -1) continue;
    next_random = next_random * (unsigned long long)25214903917 + 11;

    // LD: NEED TO MODIFY THE CODE HERE TO REPLACE THE WORD WITH TRANSLATION
    // If we decided to replace this word
    // - not based on context
    // - not already replaced by sentence replacement
    int translation = -1;
    if ((selcnt ==0) && (relcnt == 0) && (selall == 0))
    	if (join)
    		// Keep the word, replace the translation
    		translation = replace_word_with_chance(&next_random, word,NULL);
    	else
    		// Replace the word
    		word = replace_word_with_chance(&next_random, word,NULL);
    // --------------- END ---------------------


    // What is neu1 and neu1e
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

    // So that ramdomly sample the windows b
    b = next_random % window;

    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      context_size = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) { // if not the middle word
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        // LD: modify the context_size
        context_vec[context_size] = last_word;
        context_size ++;
        // Forward pass:
        // Compute hidden layer (just the average embedding of the input word which is last_word)
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }

      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw; // The actual code to get the average
        // If select the middel word based on context
        if (selcnt){
        	if (join) translation = replace_word_with_chance(&next_random,word,neu1);
        	else   word = replace_word_with_chance(&next_random,word,neu1);
        }
        else if (selall){
        	for (c =0; c < layer1_size; c++) temp_h[c] = neu1[c] + syn0[c + word * layer1_size];
        	if (join) translation = replace_word_with_chance(&next_random,word,temp_h);
        	else   word = replace_word_with_chance(&next_random,word,temp_h);
        }


        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        int no_positive = 1;
        int sense = -1;
        if ((translation != word) && (translation != -1)){
        	no_positive = 2;
        	if (join2){
            	no_positive = 3;
        		char* s_word = vocab[word].word;
        		char* t_word = vocab[translation].word;
            	char delimiter[] = ":";
        		char* sense_word;
            	if (dict_size_en[word] >0)
            		sense_word = concat(s_word,delimiter,t_word);
            	else
            		sense_word = concat(t_word,delimiter,s_word);

            	// Find whether it's in the dictionary or not
            	sense = SearchVocab(sense_word);
            	//printf ("Sense word %s and id %d \n", sense_word, sense);
        	}
        }

        int no_combine = 0;
        if (combine) no_combine = context_size;

        if (negative > 0) for (d = 0; d < negative + no_positive + no_combine; d++) {
          // Sample for getting negative sample (the first one is the correct, that why negative + 1
          if (d < no_positive){
        	  if (d == 0) {
                   target = word;
                   label = 1;
        	  }
              if ((d == 1) && (translation != -1)) { // just to be safe
               	 target = translation;
               	 label = 1;
              }
              if ((d == 2) && (sense != -1)){
            	  target = sense;
            	  label = 1;
              }

          }else
        	  if (d < no_positive + no_combine) {
        		  // Predict also the context
        		  target = context_vec[no_positive + no_combine - d - 1];
        		  label = 1;
        	  }
			  else {
				next_random = next_random * (unsigned long long)25214903917 + 11;
				target = table[(next_random >> 16) % table_size];
				if (target == 0) target = next_random % (vocab_size - 1) + 1;
				if (target == word) continue;
				if (target == translation) continue;
				if (target == sense) continue;
				int check = 0, i=0;
				if (combine) for (i =0; i<context_size; i++) if (target == context_vec[i]) {check = 1; break;}
				if (check) continue; // Ignore if this is in context
				label = 0;
			  }
          // LD: target is the randomly selected word from table

          l2 = target * layer1_size;  // l2 is the position in the embedding array where target is embedded
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          // f is the computation of equation (4) in the paper
          // Distributed Representations of Words and Phrases and their Compositionality

          // Is this similar with gradient clipping checking value of f ?
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

          // Here if we need to balance source and target (up-weight the translation)
          if ((balance) && (translation != -1) && (target == translation)) g = g / chance;
          // LD: g is the error w.r.t the learning rate.
          // This is the end of forward pass.
          // Next we compute the backward pass.

          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) {
        	  float save_value = syn1neg[c + l2];
        	  float save_value1 = syn0[c+l2];
        	  syn1neg[c + l2] += g * neu1[c];
              if (reg_sen > 0){
            	  // Must remember to put the learning rate there
            	  syn1neg[c + l2] -= 2 * alpha * reg_sen * (save_value - save_value1);
            	  // Finished update syn1neg
            	  syn0[c+l2] += 2 * alpha * reg_sen * (save_value - save_value1);
              }
          }

        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }

    } else {  //train skip-gram
      if (selcnt){
    	  // Select based on the context, first construct the context vectors

    	  for (c = 0; c < layer1_size; c++) cntvec[c] = 0;
          for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            for (c = 0; c < layer1_size; c++) cntvec[c] += syn0[c + last_word * layer1_size];
          }
          // Select the word
          if (join) translation = replace_word_with_chance(&next_random,word,cntvec);
          else   word = replace_word_with_chance(&next_random,word,cntvec);
      }
      else if(selall){
    	  printf(" SELECT ALL FOR SKIP-GRAM IS NOT IMPLEMENTED\n");
    	  exit(1);
      }

      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        int no_positive = 1;
        if (translation != -1) no_positive = 2;

        if (negative > 0) for (d = 0; d < negative + no_positive; d++) {
          if (d < no_positive){
        	  if (d == 0) {
                     target = word;
                     label = 1;
          	  }
              if ((d == 1) && (translation != -1)) { // just to be safe
                 	 target = translation;
                 	 label = 1;
              }
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            if (target == translation) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  // Print the data coverage
  printf("\nData replaced : %f", replace / (1.0 * total_words));

  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  // Read the dictionary here
  printf("Starting reading dictionary using file %s\n", dict_file);
  //printf(" Dictionary size : %d\n",vocab_size);
  ReadDict();
  if (join2){
	  printf(" Join dictionary : %d\n",vocab_size);
  }


  printf("============ Configuration =============\n");
  printf("Replace with dictionary with probability %f\n",chance);
  if (nosel) printf("RANDOMLY select from dict : \n");
  else printf("Select from dict based on DISTANCE\n");
  if (seldist) printf("Select dictionary from DISTRIBUTIOn \n");
  else printf("Select dictionary from MAX function\n");
  if (relcnt) printf("REPLACE not only the middle word but also the CONTEXT \n");

  if (selall) printf("Replace based on ALL (context + middle word)\n");
  else
	  if (selcnt) printf("Replace based on the CONTEXT\n");
	  else printf("Replace based on the MIDDLE word only\n");

  if (join2){
	  printf("Predict JOINLY the middle word, translation and SENSE\n");
	  join = 1; //
  }
  else
	  if (join) printf("Predict JOINLY the middle word and the translation\n");
	  else printf("Only predict the middle word (not joinly)\n");

  if (cbow) printf("Using CBOW model \n");
  else printf("Using SKIP-GRAM model\n");
  if (balance) printf("Using BALANCE during join training (up-weight translation)\n");
  if (combine) printf("PREDICT CONTEXT too aside from middle word and translation\n");
  if (reg_sen >0) printf("using REGULARIZATION (%f) for combining input and output embeddings\n",reg_sen);

  printf("=======================================\n");
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();

  start = clock();
  // LD: Initialize the model from reference embedding here
  if (refemb_file[0] !=0) ModelInitialization();
  // End initialization

  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    printf(" Save the word vectors \n");
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }

    // LD: Save the negative word vectors
    if (output_file_neg[0] != 0){ // not empty string
        printf(" Save the NEGATIVE word vectors \n");
        FILE* foneg = fopen(output_file_neg, "wb");
        fprintf(foneg, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {
          fprintf(foneg, "%s ", vocab[a].word);
          if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, foneg);
          else for (b = 0; b < layer1_size; b++) fprintf(foneg, "%lf ", syn1neg[a * layer1_size + b]);
          fprintf(foneg, "\n");
        }
        fclose(foneg);
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");

    printf("\t-dict <file>\n");
    printf("\t\tUse dictionary from <file> to train the cross-lingual system\n");
    printf("\t-chance <float>\n");
    printf("\t\tChance that we randomly replace with dictionary; default is 1 (100%)\n");
    printf("\t-nosel <int>\n");
    printf("\t\tSelect the dictionary or not; default is 0 (with some selections)\n");
    printf("\t-seldist <int>\n");
    printf("\t\tSelect the dictionary base on the distribution; default is 0 (~use the max function)\n");
    printf("\t-selcnt <int>\n");
    printf("\t\tSelect the dictionary base on the context; default is 0 \n");
    printf("\t-selall <int>\n");
    printf("\t\tSelect the dictionary base on the context + word; default is 1 \n");

    printf("\t-balance <int>\n");
    printf("\t\tBalance during join training so that translation is up-weighted (1/chance); default is 0 \n");


    printf("\t-join <int>\n");
    printf("\t\tJoinly predict not only middle word but also the translation; default is 1 \n");

    printf("\t-join2 <int>\n");
    printf("\t\tJoinly predict the middle word, the translation, and the sense (word_translation); default is 0 \n");

    printf("\t-combine <int>\n");
    printf("\t\tCombine CBOW with SKIP, use the context also to predict the context; default is 0\n");

    printf("\t-relcnt <int>\n");
    printf("\t\tReplace not only the middle word but also the context. default is 0 (~not replace context)\n");

    printf("\t-refemb <file>\n");
    printf("\t\tUse the reference embedding from <file> to initialize the model\n");

    printf("\t-outputn <file>\n");
    printf("\t\tUse <file> to save the resulting of negative word vectors\n");

    printf("\t-reg <float>\n");
    printf("\t\tSet the regularization sensitivity for combining input and output embeddings\n");



    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-dict", argc, argv)) > 0) strcpy( dict_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-refemb", argc, argv)) > 0) strcpy(refemb_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-chance", argc, argv)) > 0) chance = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-nosel", argc, argv)) > 0) nosel = atoi(argv[i + 1]);

  if ((i = ArgPos((char *)"-selall", argc, argv)) > 0) selall = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-seldist", argc, argv)) > 0) seldist = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-selcnt", argc, argv)) > 0) selcnt = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-relcnt", argc, argv)) > 0) relcnt = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-join", argc, argv)) > 0) join = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-join2", argc, argv)) > 0) join2 = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-balance", argc, argv)) > 0) balance = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-combine", argc, argv)) > 0) combine = atoi(argv[i + 1]);


  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);

  if ((i = ArgPos((char *)"-outputn", argc, argv)) > 0) strcpy(output_file_neg, argv[i + 1]);
  if ((i = ArgPos((char *)"-reg", argc, argv)) > 0) reg_sen = atof(argv[i + 1]);


  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
