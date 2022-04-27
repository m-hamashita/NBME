#!/bin/bash
echo $1
echo $2
kaggle datasets init -p $1
echo "{\"title\": \"$2\", \"id\": \"mpeg31/$2\", \"licenses\": [    {\"name\": \"CC0-1.0\"    }]}" > "$1/dataset-metadata.json"
kaggle datasets create -p $1 --dir-mode "zip"
