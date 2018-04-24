import os
import glob

path = './second_responses/responses/'
dest = './second_responses/marked/'
files = glob.glob(os.path.join(path, '*.txt'))

for filename in files:
    with open(filename) as f:
        filename = filename.split('/')[-1]
        fout = open(os.path.join(dest, filename), 'w')
        index = 0
        for line in f:
            line = line.strip()
            fields = line.split()
            mention = fields[-1]
            n_starts = len([x for x in mention if x=='('])
            if n_starts == 0:
                fout.write('\t' + line +'\n')
            else:
                marker = ''
                for i in range(n_starts):
                    if i == 0:
                        marker += str(index)
                    else:
                        marker += ',' + str(index)
                    index += 1
                fout.write(marker + '\t' + line +'\n')
        fout.close()
