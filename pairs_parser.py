# new_pairs = []
# first = []
# with open('pairs.txt', 'r') as file:
#     for i, line in enumerate(file):
#         splits = line.split(' ')
#         if splits[0] not in first:
#             first.append(splits[0])
#             new_pairs.append(line)
#
# with open('best_data.txt','w') as file:
#     for line in new_pairs:
#         file.write(line)

# import random
#
# lines = open('best_data.txt','r').read().split('\n')
# random.shuffle(lines)
#
# with open('best_data_shuffle.txt', 'w') as file:
#      for line in lines:
#          file.write(line + '\n')
