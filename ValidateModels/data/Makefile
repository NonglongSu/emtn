#default: 

JC: jc.fasta jc.1.fasta jc.2.fasta jc
HKY: hky.fasta hky.1.fasta hky.2.fasta hky
K2P: k2p.fasta k2p.1.fasta k2p.2.fasta k2p
TN: tn.fasta tn.1.fasta tn.2.fasta tn

.PHONY: clean 

#run dawg and save output
%.fasta: %.dawg
	dawg $< > $@

#generate frequencies matrix from sequences
#	sed gets odd/even lines then grep gets only DNA sequences
%.1.fasta: %.fasta
	sed -n '3~3p' $< | grep -o '\ [ACGT]\+' > $@
%.2.fasta: %.fasta 
	sed -n '4~3p' $< | grep -o '\ [ACGT]\+' > $@

#run HKY model with dawg output
jc: jc.1.fasta jc.2.fasta
	python3 ../JC.py jc.1.fasta jc.2.fasta 

hky: hky.1.fasta hky.2.fasta
	python3 ../HKY.py hky.1.fasta hky.2.fasta
	make -s clean

k2p: k2p.1.fasta k2p.2.fasta
	python3 ../K2P.py k2p.1.fasta k2p.2.fasta

tn: tn.1.fasta tn.2.fasta
	python3 ../TN.py tn.1.fasta tn.2.fasta
	#make -s clean

#clean files
clean:
	rm -rf *.fasta ../__pycache__
