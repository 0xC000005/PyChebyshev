CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c99

reader: reader.c
	$(CC) $(CFLAGS) -o reader reader.c -lm

clean:
	rm -f reader

.PHONY: clean
