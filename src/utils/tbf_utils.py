#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019/12/3

doc_tex = """#BeginOfDocument ENG_NW_001278_20130418_F00012ERM
rich_ere        ENG_NW_001278_20130418_F00012ERM        E599    148,156 violence        Conflict_Attack Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E619    280,288 violence        Conflict_Attack Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E639    413,421 violence        Conflict_Attack Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E737    558,565 violent Conflict_Attack Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E847    1131,1139       violence        Conflict_Attack Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E659    127,136 denounces       Contact_Broadcast       Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E681    259,268 denounces       Contact_Broadcast       Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E703    380,389 denounced       Contact_Broadcast       Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E813    856,862 called  Contact_Broadcast       Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E757    690,696 attack  Conflict_Attack Generic
rich_ere        ENG_NW_001278_20130418_F00012ERM        E784    748,753 visit   Movement_Transport-Person       Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E867    1244,1252       arrested        Justice_Arrest-Jail     Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E917    1354,1367       demonstrators   Conflict_Demonstrate    Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E946    1368,1372       took    Movement_Transport-Person       Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E1013   866,876 protestors      Conflict_Demonstrate    Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E1033   604,615 demonstrate     Conflict_Demonstrate    Generic
rich_ere        ENG_NW_001278_20130418_F00012ERM        E1057   1289,1295       attack  Conflict_Attack Actual
rich_ere        ENG_NW_001278_20130418_F00012ERM        E1086   1484,1494       questioned      Contact_Meet    Actual
@Coreference    C10     E599,E619,E639,E737,E847
@Coreference    C679    E659,E681,E703,E813
#EndOfDocument"""


def split_tbf(filename):
    lines = open(filename).readlines()
    document_list = list()
    document = None
    for line in lines:
        if line.startswith('#BeginOfDocument'):
            document = [line]
        else:
            document += [line]
        if line.startswith('#EndOfDocument'):
            document_list += [document]
            document = None

    return document_list


def load_document_dict_from_tbf(filename):
    document_list = split_tbf(filename)
    document_list = [Document.get_from_lines(document) for document in document_list]
    document_dict = {document.doc_id: document for document in document_list}
    return document_dict


class Mention:
    def __init__(self, doc_id, mention_id, offset, text, event_type, realis):
        self.doc_id = doc_id
        self.mention_id = mention_id
        self.offset = offset
        self.text = text
        self.event_type = event_type
        self.realis = realis

    @staticmethod
    def get_from_line(line):
        att = line.split('\t')
        return Mention(doc_id=att[1],
                       mention_id=att[2],
                       offset=att[3],
                       text=att[4],
                       event_type=att[5],
                       realis=att[6])

    def to_line(self):
        return "\t".join(['rich_ere', self.doc_id, self.mention_id, self.offset,
                          self.text, self.event_type, self.realis])


class Coreference:
    def __init__(self, coref_id, mention_list):
        self.coref_id = coref_id
        self.mention_list = mention_list

    @staticmethod
    def get_from_line(line):
        att = line.split('\t')
        return Coreference(coref_id=att[1],
                           mention_list=att[2].split(','))

    def to_line(self):
        return "\t".join(['@Coreference',
                          self.coref_id,
                          ','.join(self.mention_list)])

    def to_line_with_type(self, mention_dict):
        return "\t".join(['@Coreference',
                          self.coref_id,
                          ','.join(self.mention_list),
                          '\n' + ' | '.join([mention_dict[mention].event_type for mention in self.mention_list]),
                          '\n' + ' | '.join([mention_dict[mention].text for mention in self.mention_list]),
                          ])


class Document:
    def __init__(self, doc_id, mention_list, coref_list):
        self.doc_id = doc_id
        self.mention_list = mention_list
        self.coref_list = coref_list
        self.mention_dict = {mention.mention_id: mention for mention in mention_list}

    @staticmethod
    def get_from_lines(lines):
        lines = [line.strip() for line in lines]
        doc_id = lines[0].split()[1]
        mention_list = list()
        coref_list = list()
        for line in lines[1:]:
            if line.startswith("#EndOfDocument"):
                return Document(doc_id, mention_list=mention_list, coref_list=coref_list)
            elif line.startswith("@Coreference"):
                coref_list += [Coreference.get_from_line(line)]
            else:
                mention_list += [Mention.get_from_line(line)]

    def delete_mention_in_doc(self, mention_id):
        self.delete_mention_in_coref(mention_id)
        to_delete = -1
        for index, mention in enumerate(self.mention_list):
            if mention.mention_id == mention_id:
                to_delete = index
                break
        if to_delete >= 0:
            self.mention_list.pop(to_delete)

    def delete_mention_in_coref(self, mention_id):
        to_delete = -1
        for coref in self.coref_list:
            for index, mention_id_in_coref in enumerate(coref.mention_list):
                if mention_id_in_coref == mention_id:
                    to_delete = index
                    break
            if to_delete >= 0:
                coref.mention_list.pop(to_delete)
                break

        to_delete = -1
        for index, coref in enumerate(self.coref_list):
            if len(coref.mention_list) == 0:
                to_delete = index
                break
        if to_delete >= 0:
            self.coref_list.pop(to_delete)

    def to_lines(self):
        result = list()
        result += ['#BeginOfDocument %s' % self.doc_id]
        writed_mention_set = set()
        for mention in self.mention_list:
            if mention.mention_id in writed_mention_set:
                continue
            writed_mention_set.add(mention.mention_id)
            result += [mention.to_line()]
        for coref in self.coref_list:
            if len(coref.mention_list) == 1:
                continue
            result += [coref.to_line()]
        result += ["#EndOfDocument"]
        return '\n'.join(result)

    def to_lines_with_type(self):
        result = list()
        result += ['#BeginOfDocument %s' % self.doc_id]
        writed_mention_set = set()
        for mention in self.mention_list:
            if mention.mention_id in writed_mention_set:
                continue
            writed_mention_set.add(mention.mention_id)
            result += [mention.to_line()]
        for coref in self.coref_list:
            if len(coref.mention_list) == 1:
                continue
            result += [coref.to_line_with_type(self.mention_dict)]
        result += ["#EndOfDocument"]
        return '\n'.join(result)


if __name__ == "__main__":
    pass
