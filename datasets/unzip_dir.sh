#!/bin/sh

# Choose one path to compress videos
folder='./youtube-vos/JPEGImages/'
# folder='./datasets/youtube-vos/JPEGImages'

if  [ -d $folder ];then
    for file in $folder/*
    do
        if [[ "$file" != *.zip ]]
        then
            echo $file is file
        else
            echo decompressing "$file" ...
            unzip "$file" -d "${file%.*}"
            rm -rf "$file"
        fi
    done
else
    echo '['$folder']' 'is not exist. Please check the directory.'
fi

echo 'Done!'
