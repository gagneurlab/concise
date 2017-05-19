def vcf_to_seq(vcf_path, fasta_path):
    pass


def effect_from_model(model, ref, ref_rc, alt, alt_rc, methods, mutation_positions, out_annotation_all_outputs,
                      extra_args=None, **argv):
    """Convenience function to execute multiple effect predictions in one call

    # Arguments
        model: Keras model
        ref: Input sequence with the reference genotype in the mutation position
        ref_rc: Reverse complement of the 'ref' argument
        alt: Input sequence with the alternative genotype in the mutation position
        alt_rc: Reverse complement of the 'alt' argument
        methods: A list of prediction functions to be executed, e.g.: from concise.effects.ism.ism. Using the same
            function more often than once (even with different parameters) will overwrite the results of the
            previous calculation of that function.
        mutation_positions: Position on which the mutation was placed in the forward sequences
        out_annotation_all_outputs: Output labels of the model.
        extra_args: None or a list of the same length as 'methods'. The elements of the list are dictionaries with
            additional arguments that should be passed on to the respective functions in 'methods'. Arguments
            defined here will overwrite arguments that are passed to all methods.
        **argv: Additional arguments to be passed on to all methods, e.g,: out_annotation.

    # Returns
        Dictionary containing the results of the individual calculations, the keys are the
            names of the executed functions
    """
    assert isinstance(methods, list)
    if isinstance(extra_args, list):
        assert(len(extra_args) == len(methods))
    else:
        extra_args = [None] * len(methods)

    main_args = {"model": model, "ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc,
                 "mutation_positions": mutation_positions,
                 "out_annotation_all_outputs": out_annotation_all_outputs}

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
