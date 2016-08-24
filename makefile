CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: xlingemb 

xlingemb: xlingemb.c
	$(CC) -g xlingemb.c -o xlingemb $(CFLAGS)

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy
