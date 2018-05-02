#!/usr/bin bash

function clean_numbers() {
	sed -i.bak 's/.000000000000/./g' $1
}

function delete_every_nth_line() {
	TEMP_FILE=__temp_transfer_p04.csv
	awk "NR == 1 || NR % $2 == 2" $1 > $TEMP_FILE
	mv $TEMP_FILE $1
}

# Copy the raw data from python to the tex document.
OUTPUT_FILE=../tex/pgfplots/plot_data/p04_fr.csv
cp fletcher-reeves.csv $OUTPUT_FILE
delete_every_nth_line $OUTPUT_FILE "10"
clean_numbers $OUTPUT_FILE

cp fletcher-reeves_with_restart.csv ../tex/pgfplots/plot_data/p04_fr_with_restart.csv
clean_numbers $_
cp polak-ribiere.csv ../tex/pgfplots/plot_data/p04_pr.csv
clean_numbers $_

