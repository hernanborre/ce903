#!/bin/bash

#Reading from the file that shows the list of concepts
while IFS='' read -r IN || [[ -n "$IN" ]] ; do
	#Extracting the concept and images associated
	IFS=':' read -ra LINE <<< "$IN"
	concept=$(echo ${LINE[0]})
	images=$(echo ${LINE[1]})

	mkdir "training_set"
	#Building a folder per concept
	mkdir "training_set/"$concept 

	#Coping all the imagess associated to that Concep into its folder
	IFS=';' read -ra IMAGE <<< "$images"
	for i in "${IMAGE[@]}"; do
	    cp $i".jpg" "training_set/"$concept"/" 
	done
done < "$1"

#Removing all the images outside of a folder
#rm *.jpg
