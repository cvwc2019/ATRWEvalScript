import random
import os,sys
import json
import numpy as np
from .bboxmatch import match
from sklearn.metrics import average_precision_score

def eval_impl(anno,res,partial=False):
    random.seed(114514)
    with open(anno,'r') as f:
        anno=json.load(f)
        anno={int(k):v for k,v in anno.items()}
    with open(res,'r') as f:
        res=json.load(f)
        # TODO: split partial -- split gt
    if partial:
        keys=sorted(list(anno.keys()))
        random.shuffle(keys)
        keys=keys[:int(len(keys)*0.75)]
        temp={k:anno[k] for k in keys}
        anno=temp
        n_query_sing=sum(list(map(lambda k:len(list(filter(lambda o:o['query']=='sing',anno[k]))),anno)))
        n_query_multi=sum(list(map(lambda k:len(list(filter(lambda o:o['query']=='multi',anno[k]))),anno)))
    else:
        n_query_sing =701
        n_query_multi = 1063
    #iou match
    res=match(anno,res) #matched with eid and fid
    #dTODO modify the reid eval
    #dTODO counting query and FN

    eids,fids={},{}
    query_multi=[]
    query_sing=[]
    for obj in res['bboxs']:
        boxid,query,frame,entityid=obj['bbox_id'],obj['query'],obj['fid'],obj['eid']
        if query=='multi':
            query_multi.append(boxid)
        elif query=='sing':
            query_sing.append(boxid)
        eids[boxid]=entityid
        fids[boxid]=tuple(frame)
    #gallery loaded

    aps_sing,aps_multi=[],[]
    cmc_sing,cmc_multi = np.zeros(len(eids), dtype=np.int32),np.zeros(len(eids), dtype=np.int32)
    for ans in res['reid_result']:
        queryid,ans_ids=ans['query_id'],ans['ans_ids']
        if queryid in query_multi:
            aps=aps_multi
        elif queryid in query_sing:
            aps=aps_sing
        else:
            continue
        entityid=eids[queryid]
        gt_eids=np.asarray([eids[i] for i in ans_ids])
        pid_matches= gt_eids==entityid
        mask=exclude(entityid,fids[queryid],gt_eids,[fids[i] for i in ans_ids])
        distances=np.arange(len(ans_ids))+1.0
        distances[mask] = np.inf
        pid_matches[mask] = False
        scores = 1 / distances
        ap = average_precision_score(pid_matches, scores)
        if np.isnan(ap):
            # print()
            # print("WARNING: encountered an AP of NaN!")
            # print("This usually means a person only appears once.")
            # print("In this case, it's because of {}.".format(queryid))
            # print("I'm excluding this person from eval and carrying on.")
            # print()
            aps.append(0)
            continue
        aps.append(ap)
        k = np.where(pid_matches[np.argsort(distances)])[0][0]
        if queryid in query_multi:
            cmc_multi[k:] += 1
        else:
            cmc_sing[k:]  += 1
    mean_ap_sing = np.sum(aps_sing)/n_query_sing
    mean_ap_multi = np.sum(aps_multi)/n_query_multi
    top1_sing=cmc_sing[0]/n_query_sing
    top5_sing=cmc_sing[4]/n_query_sing
    top1_multi=cmc_multi[0]/n_query_multi
    top5_multi=cmc_multi[4]/n_query_multi
    return {'mAP(single_cam)':mean_ap_sing,
            'top-1(single_cam)':top1_sing,
            'top-5(single_cam)':top5_sing,
            'mAP(cross_cam)':mean_ap_multi,
            'top-1(cross_cam)':top1_multi,
            'top-5(cross_cam)':top5_multi,
            'mmAP':(mean_ap_sing+mean_ap_multi)/2}

def exclude(eid,fid,eids,fids):
    eids,fids=np.asarray(eids),np.asarray(fids)
    eid_match= (eids==eid)
    cid_match= np.logical_and( fid[0]==fids[:,0],
                               fid[1]==fids[:,1])
    frame_match=np.abs(fids[:,2]-fid[2])<=3
    mask = np.logical_and(cid_match, eid_match)
    mask = np.logical_and(mask, frame_match)
    junk_images= eids==-1
    mask = np.logical_or(mask, junk_images)

    return mask

def evaluate(anno, res, phase_codename, **kwargs):
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
                'public_split': eval_impl(anno, res,partial=True),
            },
        ]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output['result'] = [
            {
                'public_split': eval_impl(anno, res,partial=True)
            },
            {
                'private_split': eval_impl(anno, res),
            }
        ]
        print("Completed evaluation for Test Phase")
    return output
