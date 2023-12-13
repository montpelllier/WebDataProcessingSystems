from sentence_transformers import CrossEncoder

model = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v1')


def cal_sentence_similarity(pair_list):
    # pairs = zip(sentences1, sentences2)
    # list_pairs = list(pairs)
    # pair_list = [(sentences1, sentence) for sentence in sentences2]
    if not pair_list:
        return pair_list

    scores = model.predict(pair_list, show_progress_bar=False)
    # for i in range(len(pair_list)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(pair_list[i][0], pair_list[i][1], scores[i]))

    return scores


if __name__ == "__main__":
    list1 = ["Is Rome the capital of Italy?"]
    list2 = ["surely it is but many don’t know this fact that Italy was not always called as Italy.",
             "Before Italy came into being in 1861, it had several names including Italian Kingdom, Roman Empire and "
             "the Republic of Italy among others.",
             "If we start the chronicle back in time, then Rome was the first name to which Romans were giving credit.",
             "Later this city became known as “Caput Mundi” or the capital of the world..."]

    pair_list = [(list1[0], sentence) for sentence in list2]
    cal_sentence_similarity(pair_list)
