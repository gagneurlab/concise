
def vcf_to_seq(vcf_path, fasta_path):
    pass

def effect_from_model(model, ref, ref_rc, alt, alt_rc, methods, mutation_positions):
    """
    Predict the snp effects from a set of reference and alternative sequences. ref, alt, ref_rc, alt_rc sequences have
    to be in the same order.
    :param model: Keras model
    :param ref: Reference sequence (1-hot)
    :param alt: Alternative sequence (1-hot)
    :param ref_rc: Reference sequence (1-hot)
    :param alt_rc: Alternative sequence (1-hot)
    :param methods: Methods to use for effect prediction
    :param mutation_positions: Position of the mutation(s) per sequence
    :return: A dict of vectors with the predicted scores. Keys are method names
    """
    assert isinstance(methods, list)
    pass