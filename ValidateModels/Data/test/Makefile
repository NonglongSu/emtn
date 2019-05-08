
# display help message by default
default: help

NUM = 1000
LIST := $(shell echo {1..$(NUM)})
FASTA_FILES = $(addsuffix .fasta,$(addprefix output/,$(LIST)))
JSON_FILES = $(addsuffix .json,$(addprefix output/,$(LIST)))

.PHONY: help clean clean_json

# display help message
help:
	@echo "Usage: make [options] target"
	@echo "Targets:"
	@echo "  fisher_test      test fisher information for Tamura-Nei model"
	@echo "  help             display this message"

output/%.fasta:
	@dawg control.dawg -o $@

output/%.json: output/%.fasta
	@python3 ../../TN.py $< $@ --no_print
	@echo -e "File "$*" of "$(NUM)
	@echo -en "\e[1A"


fisher_test: $(JSON_FILES)
	@make -s clean
	@python3 fisher_chisq.py control.dawg $^

clean:
	@rm -f $(FASTA_FILES)

clean_json:
	@rm -f $(JSON_FILES)
