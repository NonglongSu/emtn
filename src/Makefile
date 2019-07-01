
# display help message by default
default: help

.PHONY: clean jc hky k2p tn help

# display help message
help:
	@echo "Usage: make {model}_{input method} [help]"
	@echo "  models:"
	@echo "    - JC:            Jukes Cantor"
	@echo "    - K2P:           Kimura 2 Parameter"
	@echo "    - HKY:           Hasegawa, Kishina, and Yano"
	@echo "    - TN:            Tamura-Nei"
	@echo "  input methods:"
	@echo "    - mult:          Randomly generate parameters from multinomial distribution"
	@echo "    - fasta:         Input fasta file. Must be named {model}.fasta"
	@echo "    - dawg:          Input dawg file. Must be named {model}.fasta"
	@echo "  help:"
	@echo "                     display help message"

#run dawg and save output
%.fasta: %.dawg
	@echo -n "Running dawg..."
	@dawg $< -o $@
	@echo "...done"

%_mult:
	@echo "Running EM with randomly generated parameters"
	@python3 -B main.py $*
	@rm -rf __pycache__

%_fasta:
	@echo "Running EM with fasta file input"
	@python3 -B main.py $* -i $*.fasta
	@rm -rf __pycache__

%_dawg: %.fasta
	@echo "Running EM with dawg file"
	@python3 -B main.py $* -i $<
	@rm -rf __pycache__


#clean files
clean:
	@rm -rf *.fasta ../__pycache__

################################################################################
# Tests                                                                        #
################################################################################

.PHONY: test.all test.jc test.hky test.k2p test.tn clean.tests

#test: jc.test hky.test k2p.test tn.test

jc.test:
	@echo "Testing Jukes Cantor"
	@python3 ../Tests/test_jc.py
	@make -s clean

tn.test:
	@echo "Testing Tamura Nei"
	@python3 ../Tests/test_tn.py
	@make -s clean

k2p.test:
	@echo "Testing Kimura 2 parameter"
	@python3 ../Tests/test_k2p.py
	@make -s clean
