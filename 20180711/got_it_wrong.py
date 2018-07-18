import os
import shutil

with open('output_summary.csv', 'r') as f, open('source_scores.csv', 'r') as g:
    output = []
    for summary, score in zip(f.readlines(), g.readlines()):
        score = score.strip().split(',')
        
        if float(score[-1])==0.0 and float(score[-2])==0.0:
            summary = summary.strip().split(',')
            name = summary[0]
            if name in os.listdir('./output'):
                label = summary[5:9]
                prediction = summary[9:]
                output.append(','.join([name] + label + prediction) + '\n')
                shutil.copytree('./output/{}'.format(name),
                                './got_it_wrong/{}'.format(name))

with open('./got_it_wrong/summary.csv', 'w') as f:
    f.writelines(output)
            
        
