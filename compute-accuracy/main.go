package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

const max_size int = 2000 // max length of strings
const N int = 1           // number of closest words
const max_w int = 50      // max length of vocabulary entries

func main() {
	args := os.Args
	var st []string
	var st1, st2, st3, st4 string
	var bestw []string = make([]string, N)
	var dist, length float64
	var bestd []float64 = make([]float64, N)
	var vec []float64 = make([]float64, max_size)
	var words, size, a, b, c, d, b1, b2, b3 int
	var threshold int = 0
	var TCN int
	var CCN, TACN, CACN, SECN, SYCN, SEAC, SYAC, QID, TQ, TQS int = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	if len(args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: ./compute-accuracy <FILE> <threshold>\nwhere FILE contains word projections, and threshold is used to reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30000)\n")
		os.Exit(0)
	}
	file_name := args[1]
	if len(args) > 2 {
		v, _ := strconv.ParseInt(args[2], 10, 64)
		threshold = int(v)
	}
	f, err := os.Open(file_name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Input file not found\n")
		os.Exit(-1)
	}
	defer f.Close()
	br := bufio.NewReader(f)
	fmt.Fscanf(br, "%d", &words)
	fmt.Fprintf(os.Stderr, "words: %d\n", words)
	if threshold != 0 {
		if words > threshold {
			words = threshold
			fmt.Fprintf(os.Stderr, "\t-> %d\n", words)
		}
	}
	fmt.Fscanf(br, "%d", &size)
	fmt.Fprintf(os.Stderr, "size: %d\n", size)
	vocab := make([]string, words)
	M := make([]float64, words*size)
	for b = 0; b < words; b++ {
		vocab[b], err = br.ReadString(' ')
		vocab[b] = strings.ToUpper(strings.Replace(vocab[b], "\n", "", -1))
		err = binary.Read(br, binary.LittleEndian, M[b*size:b*size+size])
		failOnError(err, "Cannot read input file")
		length = 0
		for a = 0; a < size; a++ {
			length += M[a+b*size] * M[a+b*size]
		}
		length = math.Sqrt(length)
		for a = 0; a < size; a++ {
			M[a+b*size] /= length
		}
	}
	scanner := bufio.NewScanner(os.Stdin)
	TCN = 0
	for {
		for a = 0; a < N; a++ {
			bestd[a] = 0
		}
		for a = 0; a < N; a++ {
			bestw[a] = ""
		}
		sf := scanner.Scan()
		if sf {
			st1 = scanner.Text()
			st = strings.Split(st1, " ")
			st1 = strings.ToUpper(st[0])
		}
		if !sf || st1 == ":" || st1 == "EXIT" {
			if TCN == 0 {
				TCN = 1
			}
			if QID != 0 {
				fmt.Printf("ACCURACY TOP1: %.2f %%  (%d / %d)\n", float64(CCN)/float64(TCN)*100, CCN, TCN)
				fmt.Printf("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n", float64(CACN)/float64(TACN)*100, float64(SEAC)/float64(SECN)*100, float64(SYAC)/float64(SYCN)*100)
			}
			QID++
			if !sf {
				break
			}
			if 1 < len(st) {
				fmt.Printf("%s:\n", st[1])
			}
			TCN = 0
			CCN = 0
			continue
		}
		if st1 == "EXIT" {
			break
		}
		st2 = strings.ToUpper(st[1])
		st3 = strings.ToUpper(st[2])
		st4 = strings.ToUpper(st[3])
		for b = 0; b < words; b++ {
			if vocab[b] == st1 {
				break
			}
		}
		b1 = b
		for b = 0; b < words; b++ {
			if vocab[b] == st2 {
				break
			}
		}
		b2 = b
		for b = 0; b < words; b++ {
			if vocab[b] == st3 {
				break
			}
		}
		b3 = b
		for a = 0; a < N; a++ {
			bestd[a] = 0
		}
		for a = 0; a < N; a++ {
			bestw[a] = ""
		}
		TQ++
		if b1 == words {
			continue
		}
		if b2 == words {
			continue
		}
		if b3 == words {
			continue
		}
		for b = 0; b < words; b++ {
			if vocab[b*max_w] == st4 {
				break
			}
		}
		if b == words {
			continue
		}
		for a = 0; a < size; a++ {
			vec[a] = (M[a+b2*size] - M[a+b1*size]) + M[a+b3*size]
		}
		TQS++
		for c = 0; c < words; c++ {
			if c == b1 {
				continue
			}
			if c == b2 {
				continue
			}
			if c == b3 {
				continue
			}
			dist = 0
			for a = 0; a < size; a++ {
				dist += vec[a] * M[a+c*size]
			}
			for a = 0; a < N; a++ {
				if dist > bestd[a] {
					for d = N - 1; d > a; d-- {
						bestd[d] = bestd[d-1]
						bestw[d] = bestw[d-1]
					}
					bestd[a] = dist
					bestw[a] = vocab[c*max_w]
					break
				}
			}
		}
		if st4 == bestw[0] {
			CCN++
			CACN++
			if QID <= 5 {
				SEAC++
			} else {
				SYAC++
			}
		}
		if QID <= 5 {
			SECN++
		} else {
			SYCN++
		}
		TCN++
		TACN++
	}
	fmt.Printf("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, float64(TQS)/float64(TQ)*100)
	os.Exit(0)
}

func failOnError(err error, msg string) {
	if err != nil {
		log.Fatalf("%s: %s", msg, err)
		panic(fmt.Sprintf("%s: %s", msg, err))
	}
}
