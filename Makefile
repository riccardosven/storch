CC=clang-10
CFLAGS=-Wall -pedantic -std=c11 -ggdb
CINCLUDES=-Isrc/


build:
	$(CC) $(CFLAGS) $(CINCLUDES) main.c -o main -lm

run: build
	./main
