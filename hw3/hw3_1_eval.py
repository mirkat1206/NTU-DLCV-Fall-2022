import csv
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Error: wrong format')
        print('\tpython3 hw2_3_eval.py <output csv filepath>')
        exit()
    
    out_csv = sys.argv[1]

    total_cnt = 0
    correct_cnt = 0
    with open(out_csv, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            fn = row[0]
            golden_label = fn[:fn.find('_')]
            pred_label = row[1]
            total_cnt += 1
            if golden_label == pred_label:
                correct_cnt += 1
            # print(fn, golden_label, pred_label)
    
    print('{:d}/{:d}. {:.2f}%'.format(correct_cnt, total_cnt, 100 * correct_cnt / total_cnt))