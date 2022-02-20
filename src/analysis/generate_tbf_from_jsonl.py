#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import argparse
import codecs
import json

kbp_label_set = set("""conflictattack
conflictdemonstrate
contactbroadcast
contactcontact
contactcorrespondence
contactmeet
justicearrestjail
lifedie
lifeinjure
manufactureartifact
movementtransportartifact
movementtransportperson
personnelelect
personnelendposition
personnelstartposition
transactiontransaction
transactiontransfermoney
transactiontransferownership""".split('\n'))


def first_upcase(word):
    return str.upper(word[0]) + word[1:]


def process_span_offset(span):
    left, right = span.split('-')
    return '%s,%s' % (int(left), int(right) + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='jsonl_path', required=True)
    parser.add_argument('-o', dest='output', required=True)
    args = parser.parse_args()
    new_tbf_list = list()
    with codecs.open(args.jsonl_path, 'r', 'utf8') as fin:
        for doc_json in fin:
            line_list = list()
            coref_list = list()
            document = json.loads(doc_json)
            doc_id = document['id']
            line_list += ['#BeginOfDocument %s' % doc_id]
            for event in document['event']:
                if '%s%s' % (event['mentions'][0]['type'], event['mentions'][0]['subtype']) not in kbp_label_set:
                    continue
                mention_id_list = list()
                for mention in event['mentions']:
                    # rich_ere	ENG_NW_001278_20130113_F00013PQC	E835	108,114	arrest	Justice_Arrest-Jail	Actual
                    line_list += ['richere\t%s\t%s\t%s\t%s\t%s\t%s' % (doc_id,
                                                                       mention['id'].replace('em-', 'E'),
                                                                       process_span_offset(mention['nugget']['span']),
                                                                       mention['nugget']['text'],
                                                                       '%s_%s' % (first_upcase(mention['type']),
                                                                                  first_upcase(mention['subtype'])),
                                                                       first_upcase(mention['realis'])),
                                  ]
                    mention_id_list += [mention['id'].replace('em-', 'E')]
                if len(mention_id_list) > 1:
                    coref_list += ['@Coreference\t%s\t%s' % (event['id'].replace('h-', 'C'), ','.join(mention_id_list))]
            line_list += coref_list
            line_list += ['#EndOfDocument']
            new_tbf_list += line_list

    with codecs.open(args.output, 'w', 'utf8') as output:
        output.write('\n'.join(new_tbf_list))


if __name__ == "__main__":
    main()
