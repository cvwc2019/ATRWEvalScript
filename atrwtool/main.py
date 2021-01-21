import argparse
import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__), '../'))
from atrwtool import detect,pose,plain,wild
model={'detect':detect,'pose':pose,'plain':plain,'wild':wild}
annos={
    'detect':'annotations/detect_tiger02_test.json',
    'pose':'annotations/pose_tiger02_test.json',
    'plain':'annotations/gt_test_plain.json',
    'wild':'annotations/gt_test_wild.json'}

def evaluate(input_file_path, task, **kwargs):
    output=model[task].evaluate(annos[task],input_file_path,'test')
    output["submission_result"] = output["result"][0]["public_split"]
    return output

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str,choices=['detect','pose','plain','wild'])
    parser.add_argument('input', type=str)
    args=parser.parse_args()
    output=evaluate(args.input,args.task)
    print(output)
