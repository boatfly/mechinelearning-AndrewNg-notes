import json
from collections import defaultdict
from collections import Counter
import pandas as pd


def get_counts(sequence):
    counts =  {} # 字典
    for x in sequence:
        if x in counts:
            counts[x]+=1
        else:
            counts[x]=1
    return counts


def get_counts2(sequence):
    counts = defaultdict(int)  # values will initialize to 0
    for x in sequence:
        counts[x] += 1
    return counts


def top_counts(count_dict, n=10):
    value_key_pairs = [(count,tz) for tz,count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]


path = 'example01.txt'

'''
-使用标准python函数操作
'''
'''
temp = open(path).readline()
print(temp)
'''

records = [json.loads(line) for line in open(path)]

print(len(records))
# print(records[:10])

time_zones = [rec['tz'] for rec in records if 'tz' in rec]

#print(time_zones[:10])

counts = get_counts(time_zones)

print(counts['America/Denver'])
print(len(time_zones))

print(top_counts(counts))

counts1 = Counter(time_zones)
top10 = counts1.most_common(10)

print(top10)

'''
-引入pandas
'''

