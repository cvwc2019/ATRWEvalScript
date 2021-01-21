import random
import os,sys
from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval
print('Updated')
import numpy as np

def eval_coco(anno,res,partial=False):
    random.seed(114514)
    annType = 'keypoints'
    cocoGt=COCO(anno)#todo
    cocoDt=cocoGt.loadRes(res)
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    if partial:
        imgIds=sorted(cocoGt.getImgIds())
        random.shuffle(imgIds)
        cocoEval.params.imgIds=imgIds[:len(imgIds)//2]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[0]

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    print("Starting Evaluation.....")
    # print("Submission related metadata:")
    # print(kwargs['submission_metadata'])

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")

        output['result'] = [
            {
                'public_split': {
                    'mAP': eval_coco(test_annotation_file, user_submission_file,partial=True),
                }
            },
        ]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output['result'] = [
            {
                'public_split': {
                    'mAP': eval_coco(test_annotation_file, user_submission_file,partial=True),
                }
            },
            {
                'private_split': {
                    'mAP': eval_coco(test_annotation_file, user_submission_file),
                }
            }
        ]
        print("Completed evaluation for Test Phase")
    return output
