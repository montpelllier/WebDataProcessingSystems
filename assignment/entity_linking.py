import numpy as np

from assignment.KB import *
from assignment.html_parser import *
from assignment.sentence_similarity_calculator import cal_sentence_similarity


def transfer_id2url(wikidata_ent_id):
    real_id = wikidata_ent_id.split('/')[-1]
    return "https://www.wikidata.org/wiki/" + real_id


def link_entity(sentences):
    entity_map = set()

    for sent in sentences:
        for ent in sent.ents:
            # print(ent.text + ":" + sent.text)
            briefs = []
            wikipedia_urls = []

            ent_ids = wikidata_query(ent.text, ent.type)
            for ent_id in ent_ids:
                ent_url = transfer_id2url(ent_id)
                ent_soup = parse_url(ent_url)
                ent_brief = get_wikidata_brief(ent_soup)
                ent_wikipedia_url = get_wikipedia_url(ent_soup)
                briefs.append(ent_brief)
                wikipedia_urls.append(ent_wikipedia_url)

            pairs = [(sent.text, b) for b in briefs]
            similarities = cal_sentence_similarity(pairs)
            if len(similarities) > 0:
                # only link if there is query result
                idx = np.argmax(similarities)
                entity_map.add((ent.text, wikipedia_urls[idx]))

    return entity_map
