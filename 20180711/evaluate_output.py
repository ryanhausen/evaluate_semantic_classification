import itertools

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def conf_matrix(labels, predictions, classes=None):
    if classes is None:
        classes = sorted(np.unique(labels))

    matrix = np.zeros([len(classes), len(classes)], dtype=np.int)
    
    for i, label in enumerate(classes):
        for j, prediction in enumerate(classes):
            print(label, prediction)
            print((labels==label).sum())
            print((predictions==prediction).sum())
            print(np.logical_and(labels==label, predictions==prediction).sum())
            matrix[i, j] = np.logical_and(labels==label, predictions==prediction).sum()

    return matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def cross_entropy(output, label):
    output = np.clip(output, 1e-12, 1)
    return -np.sum(label*np.log(output))

def top_k(output, label, k, strict=True):
    if strict:
        if (label > 0).sum() >= k:
            top_output = np.argsort(-output)[:k]
            top_label = np.argsort(-label)[:k]

            output_slots = np.zeros([4])
            output_slots[top_output] = 1

            label_slots = np.zeros([4])
            label_slots[top_label] = 1

            return (np.logical_and(output_slots, label_slots)).sum() / k
        else:
            return top_k(output, label, k-1, strict)
    else:
        top_output = np.argsort(-output)[0]
        top_label = np.argsort(-label)[:k]

        output_slots = np.zeros([4])
        output_slots[top_output] = 1

        label_slots = np.zeros([4])
        label_slots[top_label] = 1

        return int((np.logical_and(output_slots, label_slots)).sum() >= 1)

def uniform_label(labels):
    return [','.join(sorted(l.split(','))) for l in labels]

def main():
    col_name = 0
    col_label = slice(5,9)
    col_out = slice(9,13)

    with open('output_summary.csv', 'r') as f:
        data = f.readlines()
        
    with open('source_radii.csv', 'r') as f:
        radii = [l.strip().split(',') for l in f.readlines()]

    names = []
    rs = []
    loss = []
    top_1 = []
    top_2 = []
    in_top_2 = []
    reverse_top2 = []

    c_label = []
    c_output = []
    c_labels = np.array(['Spheroid', 'Disk', 'Irregular', 'Point Source'])

    c2_label = []
    c2_output = []

    for i, d in enumerate(data):
        print(i/len(data), end='\r')

        d = d.strip().split(',')
        name = d[col_name]
        
        missing = True
        for r in radii:
            if name==r[0].replace('_segmap', ''):
                rs.append(float(r[1]))
                missing = False
                break
        
        if missing:
            continue
                
        label = np.array([float(v) for v in d[col_label]])
        output = np.array([float(v) for v in d[col_out]])

        sorted_labels = np.argsort(-label)
        sorted_output = np.argsort(-output)

        c_label.append(c_labels[sorted_labels[0]])
        c_output.append(c_labels[sorted_output[0]])

        c2_label.append(','.join(c_labels[sorted_labels[:2]]))
        c2_output.append(','.join(c_labels[sorted_output[:2]]))

        names.append(name)
        loss.append(cross_entropy(output, label))
        top_1.append(top_k(output, label, 1))
        top_2.append(top_k(output, label, 2))
        in_top_2.append(top_k(output, label, 2, strict=False))
        reverse_top2.append(top_k(label, output, 2, strict=False))
        

    print('\n', len(names), len(rs))    
    with open('source_scores.csv', 'w') as f:
        for val in zip(names, loss, top_1, top_2, rs):
            str_vals = [str(v) for v in val]
            f.write(','.join(str_vals) + '\n')


    
    c_label= np.array(c_label)
    c_output = np.array(c_output)
    for morph in c_labels:
        print(morph, (c_label==morph).sum())

    
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confusion_matrix(c_label, c_output), sorted(c_labels), normalize=False)
    plt.savefig('./figs/confusion.pdf', dpi=600)

    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confusion_matrix(c2_label, c2_output),
                          sorted(np.unique(c2_label + c2_output)), normalize=False)
    plt.savefig('./figs/confusion_2.pdf', dpi=600)

    plt.figure(figsize=(10,10))
    uni_label = uniform_label(c2_label)
    uni_output = uniform_label(c2_output)
    plot_confusion_matrix(confusion_matrix(uni_label, uni_output),
                          sorted(np.unique(uni_label + uni_output)), normalize=False)
    plt.savefig('./figs/confusion_3.pdf', dpi=600)    



    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confusion_matrix(c_label, c_output), 
                          sorted(c_labels), 
                          normalize=True)
    plt.savefig('./figs/normalized_confusion.pdf', dpi=600)

    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confusion_matrix(c2_label, c2_output),
                          sorted(np.unique(c2_label + c2_output)), 
                          normalize=True)
    plt.savefig('./figs/normalized_confusion_2.pdf', dpi=600)

    plt.figure(figsize=(10,10))
    uni_label = uniform_label(c2_label)
    uni_output = uniform_label(c2_output)
    plot_confusion_matrix(confusion_matrix(uni_label, uni_output),
                          sorted(np.unique(uni_label + uni_output)), 
                          normalize=True)
    plt.savefig('./figs/normalized_confusion_3.pdf', dpi=600)

    top_1 = np.array(top_1)
    top_2 = np.array(top_2)
    in_top_2 = np.array(in_top_2)
    rs = np.array(rs)

    for morph in c_labels:
        with_morph = lambda x: np.logical_and(x, np.array(c_label)==morph)
    
        plt.figure()
        plt.title('Top 1')
        plt.hist(rs[with_morph(c_label==c_output)], color='g', alpha=0.5, label='correct')
        plt.hist(rs[with_morph(c_label!=c_output)], color='r', alpha=0.5, label='wrong')
        plt.legend()
        plt.savefig('./figs/top1_hist_{}.pdf'.format(morph), dpi=600)

        plt.figure()
        plt.title('Top 2')
        plt.hist(rs[with_morph(top_2==1.0)], color='g', alpha=0.5, label='correct')
        plt.hist(rs[with_morph(top_2==0.5)], color='b', alpha=0.5, label='partially correct')
        plt.hist(rs[with_morph(top_2==0.0)], color='r', alpha=0.5, label='wrong')
        plt.legend()
        plt.savefig('./figs/top2_hist_{}.pdf'.format(morph), dpi=600)

        plt.figure()
        plt.title('In Top 2')
        plt.hist(rs[with_morph(in_top_2==1.0)], color='g', alpha=0.5, label='correct')
        plt.hist(rs[with_morph(in_top_2==0.0)], color='r', alpha=0.5, label='wrong')
        plt.legend()
        plt.savefig('./figs/intop2_hist_{}.pdf'.format(morph), dpi=600)

    
        #plt.figure()
        #plt.title('Top 1')
        #plt.hist(rs[with_morph(top_1==1.0)], color='g', alpha=0.5, label='correct')
        #plt.hist(rs[with_morph(top_1==0.0)], color='r', alpha=0.5, label='wrong')
        #plt.legend()
        #plt.savefig('./figs/top1_hist_{}.pdf'.format(morph), dpi=600)

        #plt.figure()
        #plt.title('Top 2')
        #plt.hist(rs[with_morph(top_2==1.0)], color='g', alpha=0.5, label='correct')
        #plt.hist(rs[with_morph(top_2==0.5)], color='b', alpha=0.5, label='partially correct')
        #plt.hist(rs[with_morph(top_2==0.0)], color='r', alpha=0.5, label='wrong')
        #plt.legend()
        #plt.savefig('./figs/top2_hist_{}.pdf'.format(morph), dpi=600)

        #plt.figure()
        #plt.title('In Top 2')
        #plt.hist(rs[with_morph(in_top_2==1.0)], color='g', alpha=0.5, label='correct')
        #plt.hist(rs[with_morph(in_top_2==0.0)], color='r', alpha=0.5, label='wrong')
        #plt.legend()
        #plt.savefig('./figs/intop2_hist_{}.pdf'.format(morph), dpi=600)

    plt.figure()
    plt.title('Top 1')
    plt.hist(rs[top_1==1.0], color='g', alpha=0.5, label='correct')
    plt.hist(rs[top_1==0.0], color='r', alpha=0.5, label='wrong')
    plt.legend()
    plt.savefig('./figs/top1_hist.pdf', dpi=600)

    plt.figure()
    plt.title('Top 2')
    plt.hist(rs[top_2==1.0], color='g', alpha=0.5, label='correct')
    plt.hist(rs[top_2==0.5], color='b', alpha=0.5, label='partially correct')
    plt.hist(rs[top_2==0.0], color='r', alpha=0.5, label='wrong')
    plt.legend()
    plt.savefig('./figs/top2_hist.pdf', dpi=600)

    plt.figure()
    plt.title('In Top 2')
    plt.hist(rs[in_top_2==1.0], color='g', alpha=0.5, label='correct')
    plt.hist(rs[in_top_2==0.0], color='r', alpha=0.5, label='wrong')
    plt.legend()
    plt.savefig('./figs/intop2_hist.pdf', dpi=600)

    print('Average Loss: ', np.mean(loss))
    print('Average Top 1: ', np.mean(top_1))
    print('Average Top 2: ', np.mean(top_2))
    print('Average In Top 2: ', np.mean(in_top_2))
    print('Average Their Top 1 in our Top2: ', np.mean(reverse_top2))



if __name__=='__main__':
    main()

