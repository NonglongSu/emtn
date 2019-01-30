
# do nothing by default
default: 

.PHONY: clean jc hky k2p tn

#run dawg and save output
%.fasta: %.dawg
	@echo "Running dawg..."
	@dawg $< > $@

#generate frequencies matrix from sequences
#	sed gets odd/even lines then grep gets only DNA sequences
%.1.fasta: %.fasta
	@grep -o '\ [ACGT]\+' $< | sed -n '1~2p' | tr -d '\n '> $@
%.2.fasta: %.fasta 
	@grep -o '\ [ACGT]\+' $< | sed -n '2~2p' | tr -d '\n '> $@

## TODID: $^ means all arguments while $< its only the first one

#run HKY model with dawg output
jc: jc.1.fasta jc.2.fasta
	@echo "Running EM..."
	@python3 ../JC.py $^
	@echo "Cleaning files..."
	@make -s clean

hky: hky.1.fasta hky.2.fasta
	python3 ../HKY.py $^
	make -s clean

k2p: k2p.1.fasta k2p.2.fasta
	python3 ../K2P.py $^
	make -s clean

tn: tn.1.fasta tn.2.fasta
	python3 ../TN.py $^
	make -s clean

#clean files
clean:
	rm -rf *.fasta ../__pycache__

#################################################################
# Tests
#################################################################

.PHONY: test.all test.jc test.hky test.k2p test.tn clean.tests

distance = 0.2
root.length = 1000000
sim.reps = 2
alpha = 2.0
alpha2 = 2.0
beta = 1.0
piA = 0.3
piC = 0.2
piG = 0.3
piT = 0.2

test: jc.test hky.test k2p.test tn.test

#TODO: create .dawg file for each test

%.test:
	@echo "Testing $* ..."
	@make $* > $@
	@bash testing $@ 0.2
	@make -s clean.tests

jc.dawg:
	@echo "Creating jc.dawg..."
	@bash testing ${distance} ${root.length} ${sim.reps}

clean.tests:
	rm *.test

