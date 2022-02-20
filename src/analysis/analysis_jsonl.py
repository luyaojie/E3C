#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import sys
from collections import defaultdict


def main():
    in_filename = sys.argv[1]

    with open(in_filename) as fin:
        for line in fin:
            document = json.loads(line)
            entity_mention_dict = defaultdict(set)
            event_mention_dict = defaultdict(dict)
            mention_to_entity_dict = defaultdict(str)
            span_entity_dict = defaultdict(set)
            span_mention_dict = defaultdict(list)

            for entity in document['entity']:
                entity_mention_dict[entity['id']] = {mention['id'] for mention in entity['mentions']}
                for mention in entity['mentions']:
                    mention_to_entity_dict[mention['id']] = entity['id']

            for filler in document['filler']:
                mention_to_entity_dict[filler['id']] = filler['id']

            for event in document['event']:
                for mention in event['mentions']:
                    mention_dict = [
                        "%s_%s_%s" % (mention_to_entity_dict[argument['id']], argument['id'], argument['role'])
                        for argument in mention['arguments']]
                    mention_id = mention['id']
                    mention_subtype = mention['subtype']
                    mention_span = mention['nugget']['span']
                    mention_key = '_'.join([mention_id, mention_subtype, mention_span])
                    event_mention_dict[event['id']][mention_key] = mention_dict
                    span_entity_dict[mention_span].add(event['id'])
                    span_mention_dict[mention_span].append(mention)
            print(document['id'])

            for span in span_entity_dict:
                if len(span_entity_dict[span]) > 1:
                    for entity_id in span_entity_dict[span]:
                        print(event_mention_dict[entity_id])

            for span in span_mention_dict:
                if len(span_mention_dict[span]) > 1:
                    print(span_mention_dict[span])
            input()


if __name__ == "__main__":
    main()
