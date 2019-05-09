#!/bin/bash

# 用于统计有效行数
csv_fns=$(ls *_log.csv)
rm size.log
echo "items\tspace\tfname"

for i in ${csv_fns}
do
    size=$(cat ${i} | wc -l)
    echo "${size}\t\c"
    echo ${size} >> size.log
    du -h ${i}
done

echo "\ntotal size\t\c"
awk '{sums+=$1}END{print sums}' size.log
rm size.log

