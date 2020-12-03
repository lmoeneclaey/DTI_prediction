#!/bin/bash

filename="4_drugs.txt"
# while read -r line; do
#     name="$line"
#     echo "Name read from file - $name"
# done < "$filename"

drug=$(sed '2!d' $filename)
echo $drug
