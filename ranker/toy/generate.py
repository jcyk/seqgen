import random

with open('vocab_src', 'w') as fo:
    for i in range(5):
        fo.write(str(i)+'\n')

with open('vocab_tgt', 'w') as fo:
    for i in range(5):
        fo.write(str(i)+'\n')

mapping = range(5)
random.shuffle(mapping)

def get_instance():
    def toString(x):
        return ' '.join(str(i) for i in x)
    _len = random.randint(3, 5)
    src = [random.randint(0, 4) for j in range(_len)]

    tgt = [ mapping[x] for x in src]

    return '|'.join([toString(i) for i in [src, tgt]])

with open('train', 'w') as fo:
    for i in range(10000):
        fo.write( get_instance()+'\n')

with open('dev', 'w') as fo:
    for i in range(1000):
        fo.write( get_instance()+'\n')
