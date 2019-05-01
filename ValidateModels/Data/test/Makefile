
# display help message by default
default: help

NUM = 1000
LIST := $(shell echo {1..$(NUM)})
JSON_FILES = $(addsuffix .json,$(addprefix output/,$(LIST)))

.PHONY: help clean_fas clean_json

# display help message
help:
	@echo "Usage: make [options] target"
	@echo "Targets:"
	@echo "  fisher_test      test fisher information for Tamura-Nei model"
	@echo "  help             display this message"

output/%.fasta:
	@dawg control.dawg -o $@

output/%.json: output/%.fasta
	@python3 ../../TN.py $^ $@

fisher_test: $(JSON_FILES)
	@make -s clean_fas

clean_fas:
	@rm output/*.fasta

clean_json:
	@rm output/*.json
