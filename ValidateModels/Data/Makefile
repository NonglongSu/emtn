
# display help message by default
default: help

.PHONY: clean jc hky k2p tn help

# display help message
help:
	@echo "Usage: make [options] target"
	@echo "Targets:"
	@echo "  jc         run Jukes Cantor model"
	@echo "  k2p        run Kimura 2 Parameter model"
	@echo "  hky        run Hasegawa, Kishina, and Yano model"
	@echo "  tn         run Tamura-Nei model"
	@echo "  help       display help message"

#run dawg and save output
%.fasta: %.dawg
	@echo -n "Running dawg..."
	@dawg $< -o $@
	@echo "...done"

#run HKY model with dawg output
jc: jc.fasta
	@echo -n "Running EM: Jukes Cantor"
	@python3 ../JC.py $^
	@make -s clean

hky: hky.fasta
	@echo "Running EM: HKY"
	@python3 ../HKY.py $^
	@make -s clean

k2p: k2p.fasta
	@echo "Running EM: Kimura 2-param"
	@python3 ../K2P.py $^
	@make -s clean

tn: tn.fasta
	@echo "Running EM: Tamura-Nei"
	@python3 ../TN.py $^ data.json
	@make -s clean

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
