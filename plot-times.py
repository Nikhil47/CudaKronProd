#plot graph for time consumed to compute kronecker product on CPU vs on GPU

from sys import argv;
import matplotlib.pyplot as plt;

def lines(txt):
    '''
    Takes as input an opened file 
    Yields a stripped line from file one by one
    '''

    for line in txt:
        yield line.rstrip('\n');
    return;

script, filename = argv;

txt = open(filename);

for line in lines(txt):
    data = line.split();
    plt.plot(line[2], line[3]);

plt.show();
