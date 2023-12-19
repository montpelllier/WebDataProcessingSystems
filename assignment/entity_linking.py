from assignment.KB import *
from assignment.html_parser import *
from assignment.sentence_similarity_calculator import cal_sentence_similarity


def link_entity(entities, mentioned_sentences: list):
    if len(entities) != len(mentioned_sentences):
        raise Exception("length is not same!")

    entity_map = set()
    for i, ent in enumerate(entities):
        briefs = []
        wikipedia_urls = []

        ent_urls = wikidata_query(ent.text, ent.type)
        for ent_url in ent_urls:
            ent_soup = parse_url(ent_url)
            ent_brief = get_wikidata_brief(ent_soup)
            ent_wikipedia_url = get_wikipedia_url(ent_soup)
            briefs.append(ent_brief)
            wikipedia_urls.append(ent_wikipedia_url)

        pairs = [(mentioned_sentences[i], briefs[j]) for j in range(len(entities))]
        similarities = cal_sentence_similarity(pairs)
        idx = similarities.index(max(similarities))

        entity_map.add((ent.text, wikipedia_urls[idx]))

    return entity_map


