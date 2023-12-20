import stanza

from assignment.entity_linking import link_entity

stanza.download('en')  # download English model
# initialize English neural pipeline
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,mwt,pos,lemma,sentiment', download_method=None)

# text = "Trump is a Republican."
#
# doc = nlp(text)
#
# for sentence in doc.sentences:
#     for entity in sentence.ents:
#         print(f"text: {entity.text}\ttype: {entity.type}.")

q = "Is Rome the capital of Italy?"
a = (
    "surely it is but many don’t know this fact that Italy was not always called as Italy. Before Italy came "
    "into being in 1861, it had several names including Italian Kingdom, Roman Empire and the Republic of "
    "Italy among others. If we start the chronicle back in time, then Rome was the first name to which Romans "
    "were giving credit. Later this city became known as “Caput Mundi” or the capital of the world...")

q_doc = nlp(q)
a_doc = nlp(a)

sentences = q_doc.sentences + a_doc.sentences

# print(type(q_doc.sentences))
# for sent in q_doc.sentences:
#     sentences.append(sent)
#
# for sent in a_doc.sentences:
#     sentences.append(sent)

# print(sentences)
ent_map = link_entity(sentences)
print(ent_map)
