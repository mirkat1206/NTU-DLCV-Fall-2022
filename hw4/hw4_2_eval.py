import csv
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Error: wrong format')
        print('\tpython3 hw4_2_test.py <output csv file>')
        exit()
    
    csvpath = sys.argv[1]
    correct = 0
    total = 0
    with open(csvpath) as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            id, fn, klass = row
            if id != 'id':
                if fn.find(klass) != -1:
                    correct += 1
                total += 1
    print('accuracy = {}/{} ({:.2f}%)'.format(correct, total, 100.0 * correct / total))