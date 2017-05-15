def vcf_to_seq(vcf_path, fasta_path):
    pass

def effect_from_model(model, ref, ref_rc, alt, alt_rc, methods, mutation_positions, out_annotation_all_outputs,
                      extra_args = None, **argv):
    """
    Predict the snp effects from a set of reference and alternative sequences. ref, alt, ref_rc, alt_rc sequences have
    to be in the same order. Additional arguments will be passed to all methods in the list.
    :param model: Keras model
    :param ref: Reference sequence (1-hot)
    :param alt: Alternative sequence (1-hot)
    :param ref_rc: Reference sequence (1-hot)
    :param alt_rc: Alternative sequence (1-hot)
    :param methods: Methods to use for effect prediction
    :param mutation_positions: Position of the mutation(s) per sequence
    :param out_annotation_all_outputs: Labels of the outputs of the model. Must match output dimension.
    :param extra_args: Arguments that should be supplied to the methods individually.
    :return: A dict of vectors with the predicted scores. Keys are method names
    """
    assert isinstance(methods, list)
    if isinstance(extra_args, list):
        assert(len(extra_args) == len(methods))
    else:
        extra_args = [None]*len(methods)

    main_args = {"model":model, "ref":ref, "ref_rc":ref_rc, "alt":alt, "alt_rc":alt_rc,
                     "mutation_positions":mutation_positions,
                     "out_annotation_all_outputs":out_annotation_all_outputs}

    pred_results = {}
    for method, xargs in zip(methods, extra_args):
        if xargs is not None:
            if isinstance(xargs, dict):
                for k in argv:
                    if k not in xargs:
                        xargs[k] = argv[k]
        else:
            xargs = argv
        for k in main_args:
            xargs[k] = main_args[k]
        res = method(**xargs)
        pred_results[method.__name__] = res

    return pred_results
