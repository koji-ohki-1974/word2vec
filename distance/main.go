package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

const max_size int = 2000 // max length of strings
const N int = 40          // number of closest words that will be shown
const max_w int = 50      // max length of vocabulary entries

func main() {
	args := os.Args
	var st1 string
	var bestw []string = make([]string, N)
	var st []string
	var dist, length float64
	var bestd []float64 = make([]float64, N)
	var vec []float64 = make([]float64, max_size)
	var words, size, a, b, c, d, cn int
	var bi []int = make([]int, 100)
	if len(args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n")
		os.Exit(0)
	}
	file_name := args[1]
	f, err := os.Open(file_name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Input file not found\n")
		os.Exit(-1)
	}
	defer f.Close()
	br := bufio.NewReader(f)
	fmt.Fscanf(br, "%d", &words)
	fmt.Fprintf(os.Stderr, "words: %d\n", words)
	fmt.Fscanf(br, "%d", &size)
	fmt.Fprintf(os.Stderr, "size: %d\n", size)
	vocab := make([]string, words)
	M := make([]float64, words*size)
	for b = 0; b < words; b++ {
		vocab[b], err = br.ReadString(' ')
		vocab[b] = vocab[b][:len(vocab[b])-1]
		vocab[b] = strings.Replace(vocab[b], "\n", "", -1)
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
	for {
		for a = 0; a < N; a++ {
			bestd[a] = 0
		}
		for a = 0; a < N; a++ {
			bestw[a] = ""
		}
		fmt.Printf("Enter word or sentence (EXIT to break): ")
		sf := scanner.Scan()
		if !sf {
			break
		}
		st1 = scanner.Text()
		if st1 == "EXIT" {
			break
		}
		b = 0
		c = 0
		st = strings.Split(st1, " ")
		cn = len(st)
		for a = 0; a < cn; a++ {
			for b = 0; b < words; b++ {
				if vocab[b] == st[a] {
					break
				}
			}
			if b == words {
				b = -1
			}
			bi[a] = b
			fmt.Printf("\nWord: %s  Position in vocabulary: %d\n", st[a], bi[a])
			if b == -1 {
				fmt.Printf("Out of dictionary word!\n")
				break
			}
		}
		if b == -1 {
			continue
		}
		fmt.Printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n")
		for a = 0; a < size; a++ {
			vec[a] = 0
		}
		for b = 0; b < cn; b++ {
			if bi[b] == -1 {
				continue
			}
			for a = 0; a < size; a++ {
				vec[a] += M[a+bi[b]*size]
			}
		}
		length = 0
		for a = 0; a < size; a++ {
			length += vec[a] * vec[a]
		}
		length = math.Sqrt(length)
		for a = 0; a < size; a++ {
			vec[a] /= length
		}
		for a = 0; a < N; a++ {
			bestd[a] = -1
		}
		for a = 0; a < N; a++ {
			bestw[a] = ""
		}
		for c = 0; c < words; c++ {
			a = 0
			for b = 0; b < len(bi); b++ {
				if bi[b] == c {
					a = 1
				}
			}
			if a == 1 {
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
					bestw[a] = vocab[c]
					break
				}
			}
		}
		for a = 0; a < N; a++ {
			fmt.Printf("%50s\t\t%f\n", bestw[a], bestd[a])
		}
	}
	os.Exit(0)
}

func failOnError(err error, msg string) {
	if err != nil {
		log.Fatalf("%s: %s", msg, err)
		panic(fmt.Sprintf("%s: %s", msg, err))
	}
}
