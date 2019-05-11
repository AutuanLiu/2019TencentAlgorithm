#!/bin/sh

# 用于统计有效行数
csv_fns=$(ls *_log.csv)
tmp="size.log"

# 删除
if [ -f ${tmp} ]; then
    rm ${tmp}
fi

echo "items\tspace\tfname"

for i in ${csv_fns}
do
    size=$(cat ${i} | wc -l)
    echo "${size}\t\c"
    echo ${size} >> ${tmp}
    du -h ${i}
done

echo "\ntotal items\t\c"
awk '{sums+=$0}END{print sums}' ${tmp}

# 清理中间文件
if [ -f ${tmp} ]; then
    rm ${tmp}
fi



