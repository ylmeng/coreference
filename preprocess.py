import glob
import spacy
import os
from collections import defaultdict

def main():
    nlp = spacy.load('en')
    files = glob.glob('files/*.txt')

    for textfile in files:
        with open(textfile) as f:
            filestr = f.read().decode('utf8')
            doc = nlp(filestr)

            start_positions = defaultdict(int)
            end_positions = defaultdict(int)
            for chunk in doc.noun_chunks:
                start_positions[chunk.start] += 1
                end_positions[chunk.end-1] += 1

            filename = os.path.split(textfile)[-1]
            basename = filename.split('.')[0]
            fout = open('processed/' + filename, 'w')
            for i, token in enumerate(doc):
                if not token.text.strip():
                    fout.write('\n')
                    continue
                opened = False
                closed = False
                fout.write('\t'.join((basename, token.text, token.tag_)).encode('utf8'))
                fout.write('\t')
                if i in start_positions:
                    for _ in range(start_positions[i]):
                        if opened:
                            f.write('|')
                        fout.write('(0')  # 0 is a dummy label here
                        opened = True

                if i in end_positions:
                    for _ in range(end_positions[i]):
                        if closed:
                            fout.write('|')
                        if opened:
                            fout.write(')')
                        else:
                            fout.write('0)')  # 0 is a dummy label here
                        closed = True

                if not (opened or closed):
                    fout.write('-')
                fout.write('\n')
            fout.close()

if __name__ == '__main__':
    main()