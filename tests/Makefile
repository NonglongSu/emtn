
# display help message by default
default: help

NUM = 1000
LIST := $(shell echo {1..$(NUM)})
FASTA = $(addsuffix .fasta,$(addprefix output/,$(LIST)))
JSON_FIXED = $(addsuffix .json,$(addprefix ../output/fixed,$(LIST)))
JSON_MULT = $(addsuffix .json,$(addprefix ../output/mult,$(LIST)))
DAWG_MULT = $(addsuffix .dawg,$(addprefix output/mult,$(LIST)))

.PHONY: help clean clean_json

# display help message
help:
	@echo "Usage: make [options] target"
	@echo "Targets:"
	@echo "  fisher_test          test fisher information for TN with fixed parameters"
	@echo "  mult_fisher_test     test fisher information for TN with random parameters"
	@echo "  help             display this message"

################################################################################
#                         Test with fixed parameters                           #
################################################################################

../output/fixed%.fasta:
	@dawg dawg_examples/control.dawg -o $@

../output/fixed%.json: ../output/fixed%.fasta
	@python3 -B ../src/main.py TN -n -o $@ -i $<
	@echo "Json file "$*" of "$(NUM)"   "
	@echo -en "\e[1A"

fisher_test: $(JSON_FIXED)
	@python3 fisher_chisq.py dawg_examples/control.dawg $^ > chi_values.csv
	@echo "Running Anderson-Darling test"
	@R --vanilla --slave -e 'library(goftest);ad.test(read.csv("chi_values.csv",header=FALSE)$$V1)'
	@make -s clean

################################################################################
#            Test with parameters from a uniform distribution                  #
################################################################################

../output/mult%.json:
	@python3 -B ../src/main.py TN -t -n -o $@
	@echo "Json file "$*" of "$(NUM)"              "
	@echo -en "\e[1A"

mult_fisher_test: $(JSON_MULT)
	@R --vanilla --slave -e 'library(goftest);ad.test(read.csv("em_chi_values.csv",header=FALSE)$$V1)'
	@make -s clean

clean:
	@rm -f $(FASTA) $(JSON_MULT) $(JSON_FIXED)
	@rm -f em_chi_values.csv
	@rm -f iteration.csv
