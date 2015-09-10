package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
)

const MAX_STRING int = 60

const vocab_hash_size int = 500000000 // Maximum 500M entries in the vocabulary

//type real float64 // Precision of float numbers

type vocab_word struct {
	cn   int
	word string
}

type vocab_slice []vocab_word

func (me vocab_slice) Len() int {
	return len(me)
}

func (me vocab_slice) Less(i, j int) bool {
	return me[i].cn > me[j].cn
}

func (me vocab_slice) Swap(i, j int) {
	tmp := me[i]
	me[i] = me[j]
	me[j] = tmp
}

var train_file, output_file string
var vocab vocab_slice
var debug_mode int = 2
var min_count int = 5
var vocab_hash []int
var min_reduce int = 1
var vocab_max_size int = 10000
var vocab_size int = 0
var train_words int = 0
var threshold float64 = 100

var next_random uint64 = 1

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
func ReadWord(fin *bufio.Reader) (word string, err error) {
	var a int = 0
	var ch byte
	var buf bytes.Buffer
	for {
		ch, err = fin.ReadByte()
		if err == io.EOF {
			break
		}
		if ch == 13 {
			continue
		}
		if (ch == ' ') || (ch == '\t') || (ch == '\n') {
			if a > 0 {
				if ch == '\n' {
					fin.UnreadByte()
				}
				break
			}
			if ch == '\n' {
				word = "</s>"
				return
			} else {
				continue
			}
		}
		buf.WriteByte(ch)
		a++
	}
	if a >= MAX_STRING { // Truncate too long words
		buf.Truncate(MAX_STRING)
	}
	word = buf.String()
	return
}

// Returns hash value of a word
func GetWordHash(word string) uint {
	var hash uint = 1
	for a := 0; a < len(word); a++ {
		hash = hash*257 + uint(word[a])
	}
	hash = hash % uint(vocab_hash_size)
	return hash
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
func SearchVocab(word string) int {
	hash := GetWordHash(word)
	for {
		if vocab_hash[hash] == -1 {
			return -1
		}
		if word == vocab[vocab_hash[hash]].word {
			return vocab_hash[hash]
		}
		hash = (hash + 1) % uint(vocab_hash_size)
	}
	return -1
}

// Reads a word and returns its index in the vocabulary
func ReadWordIndex(fin *bufio.Reader) (int, error) {
	var word string
	word, err := ReadWord(fin)
	if err == io.EOF {
		return -1, err
	}
	return SearchVocab(word), nil
}

// Adds a word to the vocabulary
func AddWordToVocab(word string) int {
	var hash uint
	var length int = len(word) + 1
	if length > MAX_STRING {
		length = MAX_STRING
	}
	vocab[vocab_size].word = word
	vocab[vocab_size].cn = 0
	vocab_size++
	// Reallocate memory if needed
	if vocab_size+2 >= vocab_max_size {
		vocab_max_size += 10000
		vocab = append(vocab, make([]vocab_word, 10000)...)
	}
	hash = GetWordHash(word)
	for vocab_hash[hash] != -1 {
		hash = (hash + 1) % uint(vocab_hash_size)
	}
	vocab_hash[hash] = vocab_size - 1
	return vocab_size - 1
}

// Sorts the vocabulary by frequency using word counts
func SortVocab() {
	fmt.Fprintln(os.Stderr, "SortVocab")
	var hash uint
	// Sort the vocabulary and keep </s> at the first position
	sort.Sort(vocab[1:])
	for a := 0; a < vocab_hash_size; a++ {
		vocab_hash[a] = -1
	}
	for a := 0; a < vocab_size; a++ {
		// Words occuring less than min_count times will be discarded from the vocab
		if vocab[a].cn < min_count {
			vocab_size--
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(vocab[a].word)
			for vocab_hash[hash] != -1 {
				hash = (hash + 1) % uint(vocab_hash_size)
			}
			vocab_hash[hash] = a
		}
	}
	vocab = vocab[:vocab_size]
}

// Reduces the vocabulary by removing infrequent tokens
func ReduceVocab() {
	fmt.Fprintln(os.Stderr, "ReduceVocab")
	var b int = 0
	var hash uint
	for a := 0; a < vocab_size; a++ {
		if vocab[a].cn > min_reduce {
			vocab[b].cn = vocab[a].cn
			vocab[b].word = vocab[a].word
			b++
		} else {
			vocab[a].word = ""
		}
	}
	vocab_size = b
	for a := 0; a < vocab_hash_size; a++ {
		vocab_hash[a] = -1
	}
	for a := 0; a < vocab_size; a++ {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word)
		for vocab_hash[hash] != -1 {
			hash = (hash + 1) % uint(vocab_hash_size)
		}
		vocab_hash[hash] = a
	}
	//  fflush(stdout);
	min_reduce++
}

func LearnVocabFromTrainFile() {
	fmt.Fprintln(os.Stderr, "LearnVocabFromTrainFile")
	var word, last_word, bigram_word string
	var fin *bufio.Reader
	var i int
	var start int = 1
	for a := 0; a < vocab_hash_size; a++ {
		vocab_hash[a] = -1
	}
	f, err := os.Open(train_file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: training data file not found!\n")
		os.Exit(1)
	}
	defer f.Close()
	fin = bufio.NewReader(f)
	vocab_size = 0
	AddWordToVocab("</s>")
	for {
		word, err = ReadWord(fin)
		if err == io.EOF {
			break
		}
		if word == "</s>" {
			start = 1
			continue
		} else {
			start = 0
		}
		train_words++
		if (debug_mode > 1) && (train_words%100000 == 0) {
			fmt.Fprintf(os.Stderr, "Words processed: %dK     Vocab size: %dK  %c", train_words/1000, vocab_size/1000, 13)
			//      fflush(stdout);
		}
		i = SearchVocab(word)
		if i == -1 {
			a := AddWordToVocab(word)
			vocab[a].cn = 1
		} else {
			vocab[i].cn++
		}
		if start != 0 {
			continue
		}
		bigram_word = fmt.Sprintf("%s_%s", last_word, word)
		last_word = word
		i = SearchVocab(bigram_word)
		if i == -1 {
			a := AddWordToVocab(bigram_word)
			vocab[a].cn = 1
		} else {
			vocab[i].cn++
		}
		if float64(vocab_size) > float64(vocab_hash_size)*0.7 {
			ReduceVocab()
		}
	}
	SortVocab()
	if debug_mode > 0 {
		fmt.Fprintf(os.Stderr, "\nVocab size (unigrams + bigrams): %d\n", vocab_size)
		fmt.Fprintf(os.Stderr, "Words in train file: %d\n", train_words)
	}
}

func TrainModel() {
	fmt.Fprintln(os.Stderr, "TrainModel")
	var pa, pb, pab int = 0, 0, 0
	var oov int
	var i int
	var li, cn int = -1, 0
	var word, last_word, bigram_word string
	var score float64
	var fo *bufio.Writer
	var fin *bufio.Reader
	fmt.Fprintf(os.Stderr, "Starting training using file %s\n", train_file)
	LearnVocabFromTrainFile()
	f, _ := os.Open(train_file)
	defer f.Close()
	fin = bufio.NewReader(f)
	f, _ = os.Create(output_file)
	defer f.Close()
	fo = bufio.NewWriter(f)
	word = ""
	for {
		last_word = word
		word, err := ReadWord(fin)
		if err == io.EOF {
			break
		}
		if word == "</s>" {
			fo.WriteByte('\n')
			continue
		}
		cn++
		if (debug_mode > 1) && (cn%100000 == 0) {
			fmt.Fprintf(os.Stderr, "Words written: %dK%c", cn/1000, 13)
			//      fflush(stdout);
		}
		oov = 0
		i = SearchVocab(word)
		if i == -1 {
			oov = 1
		} else {
			pb = vocab[i].cn
		}
		if li == -1 {
			oov = 1
		}
		li = i
		bigram_word = fmt.Sprintf("%s_%s", last_word, word)
		i = SearchVocab(bigram_word)
		if i == -1 {
			oov = 1
		} else {
			pab = vocab[i].cn
		}
		if pa < min_count {
			oov = 1
		}
		if pb < min_count {
			oov = 1
		}
		if oov != 0 {
			score = 0
		} else {
			score = float64(pab-min_count) / float64(pa) / float64(pb) * float64(train_words)
		}
		if score > threshold {
			fo.WriteByte('_')
			fo.WriteString(word)
			pb = 0
		} else {
			fo.WriteByte(' ')
		}
		pa = pb
	}
}

func ArgPos(str string, args []string) int {
	var a int
	for a = 1; a < len(args); a++ {
		if str == args[a] {
			if a == len(args)-1 {
				fmt.Fprintf(os.Stderr, "Argument missing for %s\n", str)
				os.Exit(1)
			}
			return a
		}
	}
	return -1
}

func main() {
	args := os.Args
	if len(args) == 1 {
		fmt.Fprintf(os.Stderr, "WORD2PHRASE tool v0.1a\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		fmt.Fprintf(os.Stderr, "Parameters for training:\n")
		fmt.Fprintf(os.Stderr, "\t-train <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse text data from <file> to train the model\n")
		fmt.Fprintf(os.Stderr, "\t-output <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse <file> to save the resulting word vectors / word clusters / phrases\n")
		fmt.Fprintf(os.Stderr, "\t-min-count <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tThis will discard words that appear less than <int> times; default is 5\n")
		fmt.Fprintf(os.Stderr, "\t-threshold <float>\n")
		fmt.Fprintf(os.Stderr, "\t\t The <float> value represents threshold for forming the phrases (higher means less phrases); default 100\n")
		fmt.Fprintf(os.Stderr, "\t-debug <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet the debug mode (default = 2 = more info during training)\n")
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "./word2phrase -train text.txt -output phrases.txt -threshold 100 -debug 2\n\n")
		os.Exit(0)
	}
	if i := ArgPos("-train", args); i > 0 {
		train_file = args[i+1]
	}
	if i := ArgPos("-debug", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		debug_mode = int(v)
	}
	if i := ArgPos("-output", args); i > 0 {
		output_file = args[i+1]
	}
	if i := ArgPos("-min-count", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		min_count = int(v)
	}
	if i := ArgPos("-threshold", args); i > 0 {
		threshold, _ = strconv.ParseFloat(args[i+1], 64)
	}
	vocab = make([]vocab_word, vocab_max_size)
	vocab_hash = make([]int, vocab_hash_size)
	TrainModel()
	os.Exit(0)
}
