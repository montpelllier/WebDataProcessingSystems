from assignment.KB import *


def link_entity(entities):
    for ent in entities:
        candidates = wikidata_query(ent.text, ent.type)

