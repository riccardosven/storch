CC=clang-10
CFLAGS=-Wall -pedantic -std=c11 -ggdb
CINCLUDES=-Isrc/ 


build: main.c
	$(CC) $(CFLAGS) $(CINCLUDES) main.c -o main -lm


test: src/scorch.h tests/check.c tests/*
	$(CC) $(CINCLUDES) -Itests/ `pkg-config --cflags check`  tests/check.c -o test `pkg-config --libs check`


check: test
	./test


memcheck: test
	CK_FORK=no valgrind --leak-check=full ./test


run: build
	./main


clean:
	rm -rf main test


format:
	clang-format -i src/* tests/* -style=mozilla
