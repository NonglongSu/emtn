
# do nothing by default
default: help

.PHONY: clean jc hky k2p tn help

# display help message
help:
	@echo "Make commands:"
	@echo "jc		run Jukes Cantor model"
	@echo "k2p		run Kimura 2 Parameter model"
	@echo "hky		run Hasegawa, Kishina, and Yano model"
	@echo "tn		run Tamura-Nei model"
	@echo "help		display this message"

#run dawg and save output
%.fasta: %.dawg
	@echo -n "Running dawg..."
	@dawg $< > $@
	@echo "...done"

#generate frequencies matrix from sequences
#	sed gets odd/even lines then grep gets only DNA sequences
%.1.fasta: %.fasta
	@grep -o '\ [ACGT]\+' $< | sed -n '1~2p' | tr -d '\n '> $@
%.2.fasta: %.fasta
	@grep -o '\ [ACGT]\+' $< | sed -n '2~2p' | tr -d '\n '> $@

## TODID: $^ means all arguments while $< its only the first one

#run HKY model with dawg output
jc: jc.1.fasta jc.2.fasta
	@echo -n "Running EM..."
	@python3 ../JC.py $^
	@echo ".....done"
	@echo -n "Cleaning files..."
	@make -s clean
	@echo ".done"

hky: hky.1.fasta hky.2.fasta
	python3 ../HKY.py $^
	make -s clean

k2p: k2p.1.fasta k2p.2.fasta
	@echo "Running EM"
	@python3 ../K2P.py $^
	@make -s clean

tn: tn.1.fasta tn.2.fasta
	@python3 ../TN.py $^
	@make -s clean

#clean files
clean:
	@rm -rf *.fasta ../__pycache__

#################################################################
# Tests
#################################################################

.PHONY: test.all test.jc test.hky test.k2p test.tn clean.tests

#test: jc.test hky.test k2p.test tn.test

tn.test:
	@echo "Testing tn"
	@python3 ../Tests/test_tn.py
	@make -s clean

k2p.test:
	@echo "Testing k2p"
	@python3 ../Tests/test_k2p.py
	@make -s clean
