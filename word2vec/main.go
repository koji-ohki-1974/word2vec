package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const MAX_STRING int = 100
const EXP_TABLE_SIZE int = 1000
const MAX_EXP float64 = 6.
const MAX_SENTENCE_LENGTH int = 1000
const MAX_CODE_LENGTH int = 40

const SEEK_SET int = 0

const vocab_hash_size int = 30000000 // Maximum 30 * 0.7 = 21M words in the vocabulary

//type real float64 // Precision of float numbers

type vocab_word struct {
	cn      int
	point   []int
	word    string
	code    []byte
	codelen byte
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
var save_vocab_file, read_vocab_file string
var vocab vocab_slice
var binaryf int = 0
var cbow int = 1
var debug_mode int = 2
var window int = 5
var min_count int = 5
var num_threads int = 12
var min_reduce int = 1
var vocab_hash []int
var vocab_max_size int = 1000
var vocab_size int = 0
var layer1_size int = 100
var train_words int64 = 0
var word_count_actual int64 = 0
var iter int = 5
var file_size int64 = 0
var classes int = 0
var alpha float64 = 0.025
var starting_alpha float64
var sample float64 = 1e-3
var syn0 []float64
var syn1 []float64
var syn1neg []float64
var expTable []float64
var start time.Time

var hs int = 0
var negative int = 5

const table_size int = 1e8

var table []int

var m *sync.Mutex = new(sync.Mutex)

func InitUnigramTable() {
	fmt.Fprintln(os.Stderr, "InitUnigramTable")
	var train_words_pow float64 = 0
	var d1 float64
	var power float64 = 0.75
	table = make([]int, table_size)
	for a := 0; a < vocab_size; a++ {
		train_words_pow += math.Pow(float64(vocab[a].cn), power)
	}
	i := 0
	d1 = math.Pow(float64(vocab[i].cn), power) / train_words_pow
	for a := 0; a < table_size; a++ {
		table[a] = i
		if float64(a)/float64(table_size) > d1 {
			i++
			d1 += math.Pow(float64(vocab[i].cn), power) / train_words_pow
		}
		if i >= vocab_size {
			i = vocab_size - 1
		}
	}
}

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
	var hash uint = 0
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
		vocab_max_size += 1000
		vocab = append(vocab, make([]vocab_word, 1000)...)
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
	size := vocab_size
	train_words = 0
	for a := 0; a < size; a++ {
		// Words occuring less than min_count times will be discarded from the vocab
		if (vocab[a].cn < min_count) && (a != 0) {
			vocab_size--
			vocab[a].word = ""
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(vocab[a].word)
			for vocab_hash[hash] != -1 {
				hash = (hash + 1) % uint(vocab_hash_size)
			}
			vocab_hash[hash] = a
			train_words += int64(vocab[a].cn)
		}
	}
	vocab = vocab[:vocab_size+1]
	// Allocate memory for the binary tree construction
	for a := 0; a < vocab_size; a++ {
		vocab[a].code = make([]byte, MAX_CODE_LENGTH)
		vocab[a].point = make([]int, MAX_CODE_LENGTH)
	}
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

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
func CreateBinaryTree() {
	fmt.Fprintln(os.Stderr, "CreateBinaryTree")
	var min1i, min2i, pos1, pos2 int
	var point []int = make([]int, MAX_CODE_LENGTH)
	var code []byte = make([]byte, MAX_CODE_LENGTH)
	var count []int64 = make([]int64, vocab_size*2+1)
	var binaryt []int = make([]int, vocab_size*2+1)
	var parent_node []int = make([]int, vocab_size*2+1)
	for a := 0; a < vocab_size; a++ {
		count[a] = int64(vocab[a].cn)
	}
	for a := vocab_size; a < vocab_size*2; a++ {
		count[a] = 1e15
	}
	pos1 = vocab_size - 1
	pos2 = vocab_size
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for a := 0; a < vocab_size-1; a++ {
		// First, find two smallest nodes 'min1, min2'
		if pos1 >= 0 {
			if count[pos1] < count[pos2] {
				min1i = pos1
				pos1--
			} else {
				min1i = pos2
				pos2++
			}
		} else {
			min1i = pos2
			pos2++
		}
		if pos1 >= 0 {
			if count[pos1] < count[pos2] {
				min2i = pos1
				pos1--
			} else {
				min2i = pos2
				pos2++
			}
		} else {
			min2i = pos2
			pos2++
		}
		count[vocab_size+a] = count[min1i] + count[min2i]
		parent_node[min1i] = vocab_size + a
		parent_node[min2i] = vocab_size + a
		binaryt[min2i] = 1
	}
	// Now assign binary code to each vocabulary word
	for a := 0; a < vocab_size; a++ {
		b := a
		i := 0
		for {
			code[i] = byte(binaryt[b])
			point[i] = b
			i++
			b = parent_node[b]
			if b == vocab_size*2-2 {
				break
			}
		}
		vocab[a].codelen = byte(i)
		vocab[a].point[0] = vocab_size - 2
		for b = 0; b < i; b++ {
			vocab[a].code[i-b-1] = code[b]
			vocab[a].point[i-b] = point[b] - vocab_size
		}
	}
}

func LearnVocabFromTrainFile() {
	fmt.Fprintln(os.Stderr, "LearnVocabFromTrainFile")
	var word string
	var fin *bufio.Reader
	var i int
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
		train_words++
		if (debug_mode > 1) && (train_words%100000 == 0) {
			fmt.Fprintf(os.Stderr, "%dK%c", train_words/1000, 13)
			//      fflush(stdout);
		}
		i = SearchVocab(word)
		if i == -1 {
			a := AddWordToVocab(word)
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
		fmt.Fprintf(os.Stderr, "Vocab size: %d\n", vocab_size)
		fmt.Fprintf(os.Stderr, "Words in train file: %d\n", train_words)
	}
	fi, _ := os.Stat(train_file)
	file_size = fi.Size()
}

func SaveVocab() {
	fmt.Fprintln(os.Stderr, "SaveVocab")
	f, _ := os.Create(save_vocab_file)
	defer f.Close()
	fo := bufio.NewWriter(f)
	for i := 0; i < vocab_size; i++ {
		fmt.Fprintf(fo, "%s %d\n", vocab[i].word, vocab[i].cn)
	}
	fo.Flush()
}

func ReadVocab() {
	fmt.Fprintln(os.Stderr, "ReadVocab")
	var i int64 = 0
	var c byte
	var word string
	f, err := os.Open(train_file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Vocabulary file not found\n")
		os.Exit(1)
	}
	defer f.Close()
	fin := bufio.NewReader(f)
	for a := 0; a < vocab_hash_size; a++ {
		vocab_hash[a] = -1
	}
	vocab_size = 0
	for {
		word, err = ReadWord(fin)
		if err == io.EOF {
			break
		}
		a := AddWordToVocab(word)
		fmt.Fscanf(fin, "%d%c", &vocab[a].cn, &c)
		i++
	}
	SortVocab()
	if debug_mode > 0 {
		fmt.Fprintf(os.Stderr, "Vocab size: %d\n", vocab_size)
		fmt.Fprintf(os.Stderr, "Words in train file: %d\n", train_words)
	}
	fi, err := os.Stat(train_file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: training data file not found!\n")
		os.Exit(1)
	}
	file_size = fi.Size()
}

func InitNet() {
	fmt.Fprintln(os.Stderr, "InitNet")
	var next_random uint64 = 1
	syn0 = make([]float64, vocab_size*layer1_size)
	if hs != 0 {
		syn1 = make([]float64, vocab_size*layer1_size)
		for a := 0; a < vocab_size; a++ {
			for b := 0; b < layer1_size; b++ {
				syn1[a*layer1_size+b] = 0
			}
		}
	}
	if negative > 0 {
		syn1neg = make([]float64, vocab_size*layer1_size)
		for a := 0; a < vocab_size; a++ {
			for b := 0; b < layer1_size; b++ {
				syn1neg[a*layer1_size+b] = 0
			}
		}
	}
	for a := 0; a < vocab_size; a++ {
		for b := 0; b < layer1_size; b++ {
			next_random = next_random*uint64(25214903917) + 11
			syn0[a*layer1_size+b] = ((float64(next_random&0xFFFF) / float64(65536)) - 0.5) / float64(layer1_size)
		}
	}
	CreateBinaryTree()
}

func TrainModelThread(id int) {
	fmt.Fprintln(os.Stderr, "TrainModelThread")
	var a, b, d, cw, word, last_word int
	var sentence_length, sentence_position int = 0, 0
	var word_count, last_word_count int64 = 0, 0
	var sen []int = make([]int, MAX_SENTENCE_LENGTH+1)
	var l1, l2, c, target, label int
	var local_iter int = iter
	var next_random uint64 = uint64(id)
	var f, g float64
	var now time.Time
	var neu1 []float64 = make([]float64, layer1_size)
	var neu1e []float64 = make([]float64, layer1_size)
	fi, _ := os.Open(train_file)
	defer fi.Close()
	fi.Seek(file_size/int64(num_threads)*int64(id), SEEK_SET)
	br := bufio.NewReader(fi)
	for {
		if word_count-last_word_count > 10000 {
			//			word_count_actual += word_count - last_word_count
			atomic.AddInt64(&word_count_actual, word_count-last_word_count)
			last_word_count = word_count
			if debug_mode > 1 {
				now = time.Now()
				fmt.Fprintf(os.Stderr, "%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
					float64(word_count_actual)/float64(int64(iter)*train_words+1)*100,
					float64(word_count_actual)/(float64(now.Unix()-start.Unix()+1)*1000))
				//				fflush(stdout)
			}
			alpha = starting_alpha * (1 - float64(word_count_actual)/float64(int64(iter)*train_words+1))
			if alpha < starting_alpha*0.0001 {
				alpha = starting_alpha * 0.0001
			}
		}
		var err error
		if sentence_length == 0 {
			for {
				word, err = ReadWordIndex(br)
				if err == io.EOF {
					break
				}
				if word == -1 {
					continue
				}
				word_count++
				if word == 0 {
					break
				}
				// The subsampling randomly discards frequent words while keeping the ranking same
				if sample > 0 {
					var ran float64 = math.Sqrt(float64(vocab[word].cn)/(sample*float64(train_words))) + 1*(sample*float64(train_words))/float64(vocab[word].cn)
					next_random = next_random*25214903917 + 11
					if ran < float64(next_random&0xFFFF)/65536 {
						continue
					}
				}
				sen[sentence_length] = word
				sentence_length++
				if int(sentence_length) >= MAX_SENTENCE_LENGTH {
					break
				}
			}
			sentence_position = 0
		}
		if err == io.EOF || (word_count > train_words/int64(num_threads)) {
			word_count_actual += word_count - last_word_count
			local_iter--
			if local_iter == 0 {
				break
			}
			word_count = 0
			last_word_count = 0
			sentence_length = 0
			fi.Seek(file_size/int64(num_threads)*int64(id), SEEK_SET)
			br = bufio.NewReader(fi)
			continue
		}
		word = sen[sentence_position]
		if word == -1 {
			continue
		}
		for c = 0; c < layer1_size; c++ {
			neu1[c] = 0
		}
		for c = 0; c < layer1_size; c++ {
			neu1e[c] = 0
		}
		next_random = next_random*uint64(25214903917) + 11
		b = int(next_random % uint64(window))
		if cbow != 0 { //train the cbow architecture
			// in -> hidden
			cw = 0
			for a = b; a < window*2+1-b; a++ {
				if a != window {
					c = sentence_position - window + a
					if c < 0 {
						continue
					}
					if c >= sentence_length {
						continue
					}
					last_word = sen[c]
					if last_word == -1 {
						continue
					}
					for c = 0; c < layer1_size; c++ {
						neu1[c] += syn0[c+last_word*layer1_size]
					}
					cw++
				}
			}
			if cw != 0 {
				for c = 0; c < layer1_size; c++ {
					neu1[c] /= float64(cw)
				}
				if hs != 0 {
					for d = 0; d < int(vocab[word].codelen); d++ {
						f = 0
						l2 = vocab[word].point[d] * layer1_size
						// Propagate hidden -> output
						for c = 0; c < layer1_size; c++ {
							f += neu1[c] * syn1[c+l2]
						}
						if f <= -MAX_EXP {
							continue
						} else if f >= MAX_EXP {
							continue
						} else {
							f = expTable[(int)((f+MAX_EXP)*(float64(EXP_TABLE_SIZE)/MAX_EXP/2))]
						}
						// 'g' is the gradient multiplied by the learning rate
						g = (1 - float64(vocab[word].code[d]) - f) * alpha
						// Propagate errors output -> hidden
						for c = 0; c < layer1_size; c++ {
							neu1e[c] += g * syn1[c+l2]
						}
						// Learn weights hidden -> output
						for c = 0; c < layer1_size; c++ {
							syn1[c+l2] += g * neu1[c]
						}
					}
				}
				// NEGATIVE SAMPLING
				if negative > 0 {
					for d = 0; d < negative+1; d++ {
						if d == 0 {
							target = word
							label = 1
						} else {
							next_random = next_random*uint64(25214903917) + 11
							target = table[(next_random>>16)%uint64(table_size)]
							if target == 0 {
								target = int(next_random%uint64(vocab_size-1)) + 1
							}
							if target == word {
								continue
							}
							label = 0
						}
						l2 = target * layer1_size
						f = 0
						for c = 0; c < layer1_size; c++ {
							f += neu1[c] * syn1neg[c+l2]
						}
						if f > MAX_EXP {
							g = float64(label-1) * alpha
						} else if f < -MAX_EXP {
							g = float64(label-0) * alpha
						} else {
							g = (float64(label) - expTable[(int)((f+MAX_EXP)*(float64(EXP_TABLE_SIZE)/MAX_EXP/2))]) * alpha
						}
						for c = 0; c < layer1_size; c++ {
							neu1e[c] += g * syn1neg[c+l2]
						}
						for c = 0; c < layer1_size; c++ {
							syn1neg[c+l2] += g * neu1[c]
						}
					}
				}
				// hidden -> in
				for a = b; a < window*2+1-b; a++ {
					if a != window {
						c = sentence_position - window + a
						if c < 0 {
							continue
						}
						if c >= sentence_length {
							continue
						}
						last_word = sen[c]
						if last_word == -1 {
							continue
						}
						for c = 0; c < layer1_size; c++ {
							syn0[c+last_word*layer1_size] += neu1e[c]
						}
					}
				}
			}
		} else { //train skip-gram
			for a = b; a < window*2+1-b; a++ {
				if a != window {
					c = sentence_position - window + a
					if c < 0 {
						continue
					}
					if c >= sentence_length {
						continue
					}
					last_word = sen[c]
					if last_word == -1 {
						continue
					}
					l1 = last_word * layer1_size
					for c = 0; c < layer1_size; c++ {
						neu1e[c] = 0
					}
					// HIERARCHICAL SOFTMAX
					if hs != 0 {
						for d = 0; d < int(vocab[word].codelen); d++ {
							f = 0
							l2 = vocab[word].point[d] * layer1_size
							// Propagate hidden -> output
							for c = 0; c < layer1_size; c++ {
								f += syn0[c+l1] * syn1[c+l2]
							}
							if f <= -MAX_EXP {
								continue
							} else if f >= MAX_EXP {
								continue
							} else {
								f = expTable[(int)((f+MAX_EXP)*(float64(EXP_TABLE_SIZE)/MAX_EXP/2))]
							}
							// 'g' is the gradient multiplied by the learning rate
							g = (1 - float64(vocab[word].code[d]) - f) * alpha
							// Propagate errors output -> hidden
							for c = 0; c < layer1_size; c++ {
								neu1e[c] += g * syn1[c+l2]
							}
							// Learn weights hidden -> output
							for c = 0; c < layer1_size; c++ {
								syn1[c+l2] += g * syn0[c+l1]
							}
						}
					}
					// NEGATIVE SAMPLING
					if negative > 0 {
						for d = 0; d < negative+1; d++ {
							if d == 0 {
								target = word
								label = 1
							} else {
								next_random = next_random*uint64(25214903917) + 11
								target = table[(next_random>>16)%uint64(table_size)]
								if target == 0 {
									target = int(next_random%uint64(vocab_size-1)) + 1
								}
								if target == word {
									continue
								}
								label = 0
							}
							l2 = target * layer1_size
							f = 0
							for c = 0; c < layer1_size; c++ {
								f += syn0[c+l1] * syn1neg[c+l2]
							}
							if f > MAX_EXP {
								g = float64(label-1) * alpha
							} else if f < -MAX_EXP {
								g = float64(label-0) * alpha
							} else {
								g = (float64(label) - expTable[(int)((f+MAX_EXP)*(float64(EXP_TABLE_SIZE)/MAX_EXP/2))]) * alpha
							}
							for c = 0; c < layer1_size; c++ {
								neu1e[c] += g * syn1neg[c+l2]
							}
							for c = 0; c < layer1_size; c++ {
								syn1neg[c+l2] += g * syn0[c+l1]
							}
						}
					}
					// Learn weights input -> hidden
					for c = 0; c < layer1_size; c++ {
						syn0[c+l1] += neu1e[c]
					}
				}
			}
		}
		sentence_position++
		if sentence_position >= sentence_length {
			sentence_length = 0
			continue
		}
	}
}

func TrainModel() {
	fmt.Fprintln(os.Stderr, "TrainModel")
	var fo *bufio.Writer
	fmt.Fprintf(os.Stderr, "Starting training using file %s\n", train_file)
	starting_alpha = alpha
	if read_vocab_file != "" {
		ReadVocab()
	} else {
		LearnVocabFromTrainFile()
	}
	if save_vocab_file != "" {
		SaveVocab()
	}
	if output_file == "" {
		return
	}
	InitNet()
	if negative > 0 {
		InitUnigramTable()
	}
	start = time.Now()
	ch := make(chan int, num_threads)
	for a := 0; a < num_threads; a++ {
		go func(a int) {
			TrainModelThread(a)
			ch <- 0
		}(a)
	}
	for a := 0; a < num_threads; a++ {
		<-ch
	}
	f, _ := os.Create(output_file)
	defer f.Close()
	fo = bufio.NewWriter(f)
	if classes == 0 {
		// Save the word vectors
		fmt.Fprintf(fo, "%d %d\n", vocab_size, layer1_size)
		for a := 0; a < vocab_size; a++ {
			fmt.Fprintf(fo, "%s ", vocab[a].word)
			if binaryf != 0 {
				binary.Write(fo, binary.LittleEndian, syn0[a*layer1_size:(a+1)*layer1_size])
			} else {
				for b := 0; b < layer1_size; b++ {
					fmt.Fprintf(fo, "%lf ", syn0[a*layer1_size+b])
				}
			}
			fmt.Fprintf(fo, "\n")
		}
	} else {
		// Run K-means on the word vectors
		var clcn int = classes
		var iter int = 10
		var closeid int
		var centcn []int = make([]int, classes)
		var cl []int = make([]int, vocab_size)
		var closev, x float64
		var cent []float64 = make([]float64, classes*layer1_size)
		for a := 0; a < vocab_size; a++ {
			cl[a] = a % clcn
		}
		for a := 0; a < iter; a++ {
			for b := 0; b < clcn*layer1_size; b++ {
				cent[b] = 0
			}
			for b := 0; b < clcn; b++ {
				centcn[b] = 1
			}
			for c := 0; c < vocab_size; c++ {
				for d := 0; d < layer1_size; d++ {
					cent[layer1_size*cl[c]+d] += syn0[c*layer1_size+d]
				}
				centcn[cl[c]]++
			}
			for b := 0; b < clcn; b++ {
				closev = 0
				for c := 0; c < layer1_size; c++ {
					cent[layer1_size*b+c] /= float64(centcn[b])
					closev += cent[layer1_size*b+c] * cent[layer1_size*b+c]
				}
				closev = math.Sqrt(closev)
				for c := 0; c < layer1_size; c++ {
					cent[layer1_size*b+c] /= closev
				}
			}
			for c := 0; c < vocab_size; c++ {
				closev = -10
				closeid = 0
				for d := 0; d < clcn; d++ {
					x = 0
					for b := 0; b < layer1_size; b++ {
						x += cent[layer1_size*d+b] * syn0[c*layer1_size+b]
					}
					if x > closev {
						closev = x
						closeid = d
					}
				}
				cl[c] = closeid
			}
		}
		// Save the K-means classes
		for a := 0; a < vocab_size; a++ {
			fmt.Fprintf(fo, "%s %d\n", vocab[a].word, cl[a])
		}
	}
	fo.Flush()
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
	var i int
	if len(args) == 1 {
		fmt.Fprintf(os.Stderr, "WORD VECTOR estimation toolkit v 0.1c\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		fmt.Fprintf(os.Stderr, "Parameters for training:\n")
		fmt.Fprintf(os.Stderr, "\t-train <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse text data from <file> to train the model\n")
		fmt.Fprintf(os.Stderr, "\t-output <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse <file> to save the resulting word vectors / word clusters\n")
		fmt.Fprintf(os.Stderr, "\t-size <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet size of word vectors; default is 100\n")
		fmt.Fprintf(os.Stderr, "\t-window <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet max skip length between words; default is 5\n")
		fmt.Fprintf(os.Stderr, "\t-sample <float>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n")
		fmt.Fprintf(os.Stderr, "\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n")
		fmt.Fprintf(os.Stderr, "\t-hs <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse Hierarchical Softmax; default is 0 (not used)\n")
		fmt.Fprintf(os.Stderr, "\t-negative <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n")
		fmt.Fprintf(os.Stderr, "\t-threads <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse <int> threads (default 12)\n")
		fmt.Fprintf(os.Stderr, "\t-iter <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tRun more training iterations (default 5)\n")
		fmt.Fprintf(os.Stderr, "\t-min-count <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tThis will discard words that appear less than <int> times; default is 5\n")
		fmt.Fprintf(os.Stderr, "\t-alpha <float>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n")
		fmt.Fprintf(os.Stderr, "\t-classes <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n")
		fmt.Fprintf(os.Stderr, "\t-debug <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet the debug mode (default = 2 = more info during training)\n")
		fmt.Fprintf(os.Stderr, "\t-binary <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSave the resulting vectors in binary moded; default is 0 (off)\n")
		fmt.Fprintf(os.Stderr, "\t-save-vocab <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tThe vocabulary will be saved to <file>\n")
		fmt.Fprintf(os.Stderr, "\t-read-vocab <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tThe vocabulary will be read from <file>, not constructed from the training data\n")
		fmt.Fprintf(os.Stderr, "\t-cbow <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n")
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n")
		return
	}
	output_file = ""
	save_vocab_file = ""
	read_vocab_file = ""
	if i := ArgPos("-size", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		layer1_size = int(v)
	}
	if i := ArgPos("-train", args); i > 0 {
		train_file = args[i+1]
	}
	if i := ArgPos("-save-vocab", args); i > 0 {
		save_vocab_file = args[i+1]
	}
	if i := ArgPos("-read-vocab", args); i > 0 {
		read_vocab_file = args[i+1]
	}
	if i := ArgPos("-debug", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		debug_mode = int(v)
	}
	if i := ArgPos("-binary", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		binaryf = int(v)
	}
	if i := ArgPos("-cbow", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		cbow = int(v)
	}
	if cbow != 0 {
		alpha = 0.05
	}
	if i := ArgPos("-alpha", args); i > 0 {
		v, _ := strconv.ParseFloat(args[i+1], 64)
		alpha = float64(v)
	}
	if i := ArgPos("-output", args); i > 0 {
		output_file = args[i+1]
	}
	if i := ArgPos("-window", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		window = int(v)
	}
	if i := ArgPos("-sample", args); i > 0 {
		v, _ := strconv.ParseFloat(args[i+1], 64)
		sample = float64(v)
	}
	if i := ArgPos("-hs", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		hs = int(v)
	}
	if i := ArgPos("-negative", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		negative = int(v)
	}
	fmt.Fprintf(os.Stderr, "negative: %d\n", negative)
	if i := ArgPos("-threads", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		num_threads = int(v)
	}
	if i := ArgPos("-iter", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		iter = int(v)
	}
	if i := ArgPos("-min-count", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		min_count = int(v)
	}
	if i := ArgPos("-classes", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		classes = int(v)
	}
	vocab = make([]vocab_word, vocab_max_size)
	vocab_hash = make([]int, vocab_hash_size)
	expTable = make([]float64, EXP_TABLE_SIZE+1)
	for i = 0; i < EXP_TABLE_SIZE; i++ {
		expTable[i] = math.Exp((float64(i)/float64(EXP_TABLE_SIZE)*2 - 1) * MAX_EXP) // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1)                                // Precompute f(x) = x / (x + 1)
	}
	TrainModel()
}
