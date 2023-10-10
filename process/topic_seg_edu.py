import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import json
import glob
from glob import glob
from os.path import join as pjoin
import os
from src.others.tokenization import BertTokenizer

model = SentenceTransformer(r'C:\Users\王鑫\.cache\torch\sentence_transformers\sentence-transformers_all-mpnet-base-v2',
                            device='cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
logger = logging.getLogger('topic-segment')


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)


def load_dict(filename):
    '''load dict from json file'''
    with open(filename, "r", encoding='utf-8') as json_file:
        dic = json.load(json_file)
    return dic


if os.path.exists('./sim.json'):
    sim_cache = load_dict('./sim.json')
else:
    sim_cache = {}

if os.path.exists('sent_edu.json'):
    sent_edu = load_dict('sent_edu.json')
else:
    sent_edu = {}


def cos_sim(s1, s2):
    s1 = ' '.join(s1)
    s2 = ' '.join(s2)
    if (s1 + '**' + s2) in sim_cache:
        return sim_cache[(s1 + '**' + s2)]
    embedding1 = model.encode(s1, convert_to_tensor=True)
    embedding2 = model.encode(s2, convert_to_tensor=True)
    cos_score = util.cos_sim(embedding1, embedding2)
    sim_cache[(s1 + '**' + s2)] = cos_score[0][0].cpu().numpy()
    return cos_score[0][0].cpu().numpy()


# 判断是否存在句子数量<5的主题。
# 如果存在，则找到句子数最少的主题。该主题对应的下标存在两个，返回其左右主题中包含篇幅较少的主题对应的下标。
def contain_few_sents(seg_idx, seg_score_dict):
    # 原文并没有进行分割，不需要合并，直接返回。
    if len(seg_idx) == 1:
        return -1
    pre = -1
    min_sentence_number = 1000007
    # 代表以该下标结尾的主题包含的句子数量最少。下标表示的是主题的下标，不是句子下标。
    min_topic_idx = -1
    for i in range(len(seg_idx)):
        if seg_idx[i] - pre < min_sentence_number:
            min_sentence_number = seg_idx[i] - pre
            min_topic_idx = i
        pre = seg_idx[i]
    if min_sentence_number >= 5:
        return -1
    else:
        # 分三种情况讨论
        # 1.第一个主题
        # 2.最后一个主题
        # 3.处于中间状态的主题
        if min_topic_idx == 0:  # 为了方便计算，在传入参数的时候添加了该章节最后一句的下标。
            return min_topic_idx
        elif seg_idx[min_topic_idx] == seg_idx[-1]:
            return min_topic_idx - 1
        else:
            # 判断其左右主题中哪个篇幅较少，返回较少的对应下标。 4 3 5
            right_topic_len = seg_idx[min_topic_idx + 1] - seg_idx[min_topic_idx]
            left_topic_len = seg_idx[min_topic_idx - 1] - (seg_idx[min_topic_idx - 2] if min_topic_idx >= 2 else -1)
            if right_topic_len > left_topic_len:
                # 右边主题的篇幅较长，和左边的主题合并。
                return min_topic_idx - 1
            elif right_topic_len < left_topic_len:
                # 左边主题的篇幅较长，和右边的主题合并。
                return min_topic_idx
            else:
                if seg_score_dict.get(seg_idx[min_topic_idx]) < seg_score_dict.get(seg_idx[min_topic_idx - 1]):
                    return min_topic_idx
                else:
                    return min_topic_idx - 1


# 参数：句子集合，标签集合(该句为摘要，如 label = [0],则第一句为摘要)，摘要，阈值。返回值：json数组。
def topic_segmentation(src, label, tgt, threshold, sec_dis_span, sec_edu_label,is_test,test_file_name,with_section_str):
    # 移除字符数小于5的句子。
    dataset = []
    idxs = [i for i, s in enumerate(src) if len(s) > 5]
    temp_src = []
    temp_label = []
    dis_span = []
    edu_cnt = sum([len(temp) for temp in sec_dis_span])
    temp_edu_label = [0 for temp in range(edu_cnt)]
    for temp in sec_edu_label:
        temp_edu_label[temp] = 1
    edu_label = []
    cnt = 0
    for i in range(len(src)):
        if i in idxs:
            temp_src.append(src[i])
            dis_span.append(sec_dis_span[i])
            edu_label.extend(temp_edu_label[:len(sec_dis_span[i])])
            if i in label:
                temp_label.append(i - cnt)
        else:
            cnt += 1
        del temp_edu_label[:len(sec_dis_span[i])]
    # 处理成dis_span的格式。
    temp_edu_label = []
    for span in dis_span:
        temp_edu_label.append(edu_label[:len(span)])
        del edu_label[:len(span)]
    edu_label = temp_edu_label
    src = temp_src
    src = [src[i][:200] for i, sent in enumerate(src)]
    section_src_list = src
    label = temp_label
    # dis_span, edu_label是最终的数据,edu_label是[0，0，1，0，1，0]形式。
    if len(src) <= 5:
        if len(src) > 0:
            if not with_section_str:
                dataset.append({'sent': src, 'sent_label': label,
                                'tgt_list_str': [' '.join(tokens) for tokens in tgt],
                                'tgt_tok_list_list_str': tgt, 'edu_label': edu_label,
                                'disco_span': dis_span, 'disco_dep': [], 'disco_link': [], 'coref': [], 'doc_id': 'null'
                                })
            else:
                dataset.append({'sent': src, 'sent_label': label,
                                'tgt_list_str': [' '.join(tokens) for tokens in tgt],
                                'tgt_tok_list_list_str': tgt, 'edu_label': edu_label,
                                'disco_span': dis_span, 'disco_dep': [], 'disco_link': [], 'coref': [], 'doc_id': 'null',
                                'section':section_src_list
                                })
        else:
            return []
    else:
        sim_list = []
        # 只用前50个token计算相似度.
        for i in range(0, len(src) - 1):
            sim_list.append(cos_sim(src[i][:50], src[i + 1][:50]))
        # 局部极小值法主题分割
        # 1.找到相邻句相似度列表的峰值和谷值。
        min = []
        max = []
        for i in range(1, len(sim_list) - 1):
            if sim_list[i] < sim_list[i - 1] and sim_list[i] < sim_list[i + 1]:
                min.append(i)
            if sim_list[i] > sim_list[i - 1] and sim_list[i] > sim_list[i + 1]:
                max.append(i)
        # 避免出现谷值左右没有峰值的情况，应该是正确的。
        max.insert(0, 0)
        max.append(len(sim_list) - 1)
        # 2.计算每个谷值的深度得分,得到分割点。
        # seg_idx = [6,14]，6，14分别是第一和第二主题边界句的下标。
        seg_idx = []
        seg_score_dict = {}
        for i in min:
            left = -1
            right = -1
            for j in max:
                if j < i:
                    left = j
                if j > i:
                    right = j
                    break
            if left != -1 and right != -1:
                depth_score = (sim_list[left] + sim_list[right]) / (2 * sim_list[i]) - 1
                if depth_score > threshold:
                    seg_idx.append(i)
                    seg_score_dict[i] = depth_score

        # 新的合并规则
        while True:
            remove_idx = contain_few_sents(seg_idx + [len(src) - 1], seg_score_dict)
            if remove_idx >= 0:
                del seg_idx[remove_idx]
            else:
                break
        # 3.进行分割，返回json数据。
        pre_seg_idx = 0
        cnt_sent = 0
        # 分割之后需要处理label的数值.
        for i in seg_idx:
            topic_src = src[pre_seg_idx:i + 1]
            topic_dis_span = dis_span[pre_seg_idx:i + 1]
            topic_edu_label = edu_label[pre_seg_idx:i + 1]
            pre_seg_idx = i + 1
            topic_label = []
            for j in label:
                if cnt_sent <= j < cnt_sent + len(topic_src):
                    topic_label.append(j - cnt_sent)
            cnt_sent += len(topic_src)
            if not with_section_str:
                dataset.append({'sent': topic_src, 'sent_label': topic_label,
                                'tgt_list_str': [' '.join(tokens) for tokens in tgt],
                                'tgt_tok_list_list_str': tgt, 'edu_label': topic_edu_label,
                                'disco_span': topic_dis_span, 'disco_dep': [], 'disco_link': [], 'coref': [], 'doc_id': 'null'
                                })
            else:
                dataset.append({'sent': topic_src, 'sent_label': topic_label,
                                'tgt_list_str': [' '.join(tokens) for tokens in tgt],
                                'tgt_tok_list_list_str': tgt, 'edu_label': topic_edu_label,
                                'disco_span': topic_dis_span, 'disco_dep': [], 'disco_link': [], 'coref': [], 'doc_id': 'null',
                                'section': section_src_list
                                })

        # 最后一部分未处理
        topic_src = src[pre_seg_idx:]
        topic_dis_span = dis_span[pre_seg_idx:]
        topic_edu_label = edu_label[pre_seg_idx:]
        topic_label = []
        for j in label:
            if cnt_sent <= j <= cnt_sent + len(topic_src):
                topic_label.append(j - cnt_sent)
        if not with_section_str:
            dataset.append({'sent': topic_src, 'sent_label': topic_label,
                            'tgt_list_str': [' '.join(tokens) for tokens in tgt],
                            'tgt_tok_list_list_str': tgt, 'edu_label': topic_edu_label,
                            'disco_span': topic_dis_span, 'disco_dep': [], 'disco_link': [], 'coref': [], 'doc_id': 'null'
                            })
        else:
            dataset.append({'sent': topic_src, 'sent_label': topic_label,
                            'tgt_list_str': [' '.join(tokens) for tokens in tgt],
                            'tgt_tok_list_list_str': tgt, 'edu_label': topic_edu_label,
                            'disco_span': topic_dis_span, 'disco_dep': [], 'disco_link': [], 'coref': [], 'doc_id': 'null',
                            'section':section_src_list
                            })
        # change sent from list to dict
    for d in dataset:
        sent_list = []
        sents = d['sent']
        for idx, sent in enumerate(sents):
            sent_dict = {}
            sent_dict['sent_id'] = idx
            sent_dict['tokens'] = sent
            sent_dict['parse'] = 'null'
            sent_dict['corefs'] = []
            sent_list.append(sent_dict)
        d['sent'] = sent_list
        if is_test:
            d['doc_id'] = test_file_name
    # process section as sent.
    if with_section_str:
        for d in dataset:
            sect_sents_list = []
            sect_sents = d['section']
            for idx, sent in enumerate(sect_sents):
                sec_sent_dict = {}
                sec_sent_dict['sent_id'] = idx
                sec_sent_dict['tokens'] = sent
                sec_sent_dict['parse'] = 'null'
                sec_sent_dict['corefs'] = []
                sect_sents_list.append(sec_sent_dict)
            d['section'] = sect_sents_list

    return dataset


def split(file_lst, save_path, threshold, corpus_type,with_section_str,with_pos_info):
    file_idx = -1
    test_file_name = 0
    for file in file_lst:
        json_file = json.load(open(file, encoding='utf-8'))
        dataset = []
        file_idx += 1
        print("cur", file_idx, 'total', len(file_lst))
        for idx, doc in enumerate(json_file):
            source, sections, sent_labels, tgt = doc['src'], doc['sections'], doc['label'], doc['tgt']
            edu, edu_map_to_sent, edu_label = doc['edu'], doc['edu_map_to_sent'], doc['edu_label']
            # 过滤部分数据，
            if corpus_type != 'test':
                tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
                    [' '.join(tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=False))
                     for tt in tgt]) + ' [unused1]'
                tgt_subtoken = tgt_subtokens_str.split()[:5000]
                if len(tgt_subtoken) < 50:
                    continue
            if corpus_type != 'test' and len(source) < 20:
                continue

            last_section = sections[-1]
            # 先按照章节划分，再按照主题划分,cnt用来计算之前的句子数目，保证txt,src可以对应。
            cnt = 0
            edu_cnt = 0
            for sec_idx in range(1, last_section + 1):
                sec_txt = []
                sec_label = []
                sec_edu_text = []
                sec_edu_label = []
                # dis_span:each sentence has its dis_span(list)
                # sentence,dis_soan = [[begin_token_pos,end_token_pos](edu1),[begin_token_pos,end_token_pos](edu2)]
                sec_dis_span = []
                for i in range(len(sections)):
                    if sections[i] == sec_idx:
                        edu_map_sent = edu_map_to_sent[i]
                        temp_edu_text = []
                        sec_txt.append(source[i])
                        sec_dis_span.append([])
                        if i in sent_labels:
                            sec_label.append(i - cnt)
                        end_idx = -1
                        for idx in range(edu_map_sent[0], edu_map_sent[1]):
                            # 仔细研究下到底是一级列表还是要二级列表。决定要不要sec_edu_map_sent.
                            temp_edu_text.append(edu[idx])
                            if idx in edu_label:
                                sec_edu_label.append(idx - edu_cnt)
                            begin_idx = end_idx + 1
                            end_idx = begin_idx + len(edu[idx]) - 1
                            if len(edu[idx]) > 50:
                                print('edu_len > 50:cur_len {}'.format(len(edu[idx])))
                            sec_dis_span[-1].append([begin_idx, end_idx])
                        sec_edu_text.append(temp_edu_text)
                    elif sections[i] > sec_idx:
                        break
                cnt += len(sec_txt)
                edu_cnt += sum([len(s) for s in sec_edu_text])
                sub_data = topic_segmentation(sec_txt, sec_label, tgt, threshold, sec_dis_span, sec_edu_label,
                                              corpus_type == 'test',test_file_name,with_section_str)
                if with_pos_info:
                    for top_idx,item in enumerate(sub_data):
                        nedu = sum([len(d) for d in item['edu_label']])
                        item['section_pos'] = [sec_idx]*nedu
                        item['topic_pos'] = [top_idx + 1] * nedu
                        dataset.append(item)
                else:
                    for i, item in enumerate(sub_data):
                        dataset.append(item)
            if corpus_type == 'test':
                with open(save_path + "\\test." + str(test_file_name) + ".json", 'w') as save:
                    save.write(json.dumps(dataset))
                test_file_name += 1
                dataset = []
        if corpus_type == 'train':
            real_name = file.split('\\')[-1]
            with open(save_path + "\\" + real_name, 'w') as save:
                save.write(json.dumps(dataset))


def get_all_file(raw_path, save_path, threshold,with_section_str,with_pos_info):
    datasets = ['train', 'test']
    # datasets = ['train']
    for corpus_type in datasets:
        file_lst = []
        for json_f in glob(pjoin(raw_path, '*' + corpus_type + '.*.json')):
            file_lst.append(json_f)
        split(file_lst, save_path, threshold, corpus_type,with_section_str,with_pos_info)


def sent_txt():
    path = r'D:\DataSet\PreSumm-master\data\sent'
    sent_list = []
    for txt in os.listdir(path):
        if not txt.endswith('txt'):
            continue
        with open(path + '/' + txt, 'r', encoding='utf-8') as f:
            sent_list.extend([' '.join(sent.strip().split(' ')[:200]) for sent in f.readlines()])
    for i in range(126):
        with open(path + '/' + 'sent_list_' + str(i) + '.txt', 'w', encoding='utf-8') as f:
            for sent in sent_list[i * 10000:(i + 1) * 10000]:
                f.write(sent + '\n')


# max 1347 0.5
# max 1931 1.0
# max section 7000+,only 1% > 2048
def analysis_topic_length(path='../json_data_edu'):
    doc_list = []
    for file in os.listdir(path):
        if not file.endswith('json'):
            continue
        f = json.load(open(os.path.join(path, file)))
        for idx, doc in enumerate(f):
            print('{},{}'.format(file, idx))
            doc_list.append(sum([len(sent['tokens']) for sent in doc['sent']]))
    doc_list = sorted(doc_list,reverse=True)
    print(doc_list[0])


def analysis_error():
    path = '../bert_data'
    for file in os.listdir(path):
        if file != 'train.43.bert.pt':
            continue
        f = torch.load(os.path.join(path,file))
        for doc in f:
            sent = doc['sent_txt']
            if ' '.join(sent[0]).__contains__('given a transcript of token courses t = {'):
                print(file)
    pass

# sent_ntoken:29.43,edu_ntoken:8.36,slides_sent_ntoken:44.86
# slides_sent_ntoken有明显问题。
def data_static(path):
    paper_sent_list = []
    paper_edu_list = []
    slides_sent_list = []
    num = 0
    for file in os.listdir(path):
        json_path = os.path.join(path,file)
        f = json.load(open(json_path,encoding='utf-8'))
        num += len(f)
        for p in f:
            paper_sent_list.extend([s[:200] for s in p['src']])
            paper_edu_list.extend([s[:50] for s in p['edu']])
            slides_sent_list.extend([s for s in p['tgt']])
    paper_token = sum([len(_)for _ in paper_sent_list])
    slides_token = sum([len(_)for _ in slides_sent_list])
    paper_edu_token = sum([len(_)for _ in paper_edu_list])
    paper_snet_num = len(paper_sent_list)
    paper_edu_num = len(paper_edu_list)
    slides_sent_num = len(slides_sent_list)
    paper_snet_ntoken = paper_token / paper_snet_num
    slides_sent_ntoken = slides_token / slides_sent_num
    paper_edu_ntoken = paper_edu_token / paper_edu_num
    print(paper_snet_ntoken,paper_edu_ntoken,slides_sent_ntoken)
    # paper_edu_ntoken = paper_edu_token / paper_edu_num


if __name__ == '__main__':
    # analysis_topic_length()
    # analysis_error()
    # data_static(r'../label_json_data_edu')
    raw_path = r'../label_json_data_edu'
    save_path = r'../json_data_edu'
    threshold = 1.0
    with_section_str = False
    with_pos_info = True
    get_all_file(raw_path, save_path, threshold,with_section_str,with_pos_info)
    # analysis_topic_length()
    # 计算doc_n_edu。
    # doc_list = []
    # cnt = 0
    # for file in os.listdir(raw_path):
    #     f = json.load(open(os.path.join(raw_path,file),encoding='utf-8'))
    #     for doc in f:
    #         sent_list = []
    #         cnt += 1
    #         for sent in doc['src']:
    #             s = ' '.join(sent)
    #             sent_list.append(s)
    #         doc_list.append(sent_list)
    # print(len(doc_list))
    # sent_edu_dict = load_dict('./sent_edu.json')
    # edu_list = []
    # for doc in doc_list:
    #     temp = []
    #     for sent in doc:
    #         if len(sent.split(' ')) <= 5:
    #             edus = [sent]
    #         else:
    #             edus = sent_edu_dict[sent]
    #         temp.extend(edus)
    #     edu_list.append(temp)
# import torch
# data = torch.load(r'D:\DataSet\SciBERTSUM\bert_data\train.78.bert.pt',map_location={'cuda:1':'cuda:0'})
# print(data)
#
# import json
# path = r'D:\DataSet\PreSumm-master\json_data\test.0.json'
# save_path = r'D:\DataSet\SciBERTSUM\json_copy\copy.train.78.json'
# file = json.load(open(path))
# dataset = []
# for idx,doc in enumerate(file):
#     src = doc['src']
#     if len(src) == 0:
#         print(doc['label'])
#     section = doc['sections']
#     tgt = doc['tgt']
#     label = [0 * 10]
#     dic = {'src':src,'section':section,'tgt':tgt,'label':label}
#     dataset.append(dic)

# with open(save_path,'w') as save:
#     save.write(json.dumps(dataset))
# print(file)
