def import_pipeline_components(
    include_categorical: bool = False,
    include_numerical: bool = True,
    include_string: bool = False,
    multiclass_classification: bool = False) -> None:

    # loading and splitting

    from implementations.load_data.load_csv_data import LoadCSVData

    # category coalescence
    from implementations.category_coalescence.no_category_coalescence import NoCategoryCoalescence

    # encoding
    from implementations.encoding.no_encoding import NoEncoding

    # imputing
    from implementations.numerical_imputers.simple_imputer import SKLSimpleImpute

    # scaling
    from implementations.scalers.minmax_scaler import SKLMinMaxScaler
    from implementations.scalers.quantile_transformer import SKLQuantileTransformer
    from implementations.scalers.robust_scaler import SKLRobustScaler
    from implementations.scalers.standared_scaler import SKLStandardScaler
    from implementations.scalers.normalizer import SKLNormalizer
    from implementations.scalers.power_transformer import SKLPowerTransformer
    from implementations.scalers.no_scaling import NoScaling

    # feature preprocessing
    from implementations.feature_preprocessors.fast_ica import SKLFastICA
    from implementations.feature_preprocessors.feature_agglomeration import SKLFeatureAgglomeration
    from implementations.feature_preprocessors.kernel_pca import SKLKernelPCA
    from implementations.feature_preprocessors.no_feature_preprocessor import NoFeaturePreprocessor
    from implementations.feature_preprocessors.nystroem import SKLNystroem
    from implementations.feature_preprocessors.pca import SKLPCA
    from implementations.feature_preprocessors.polynomial_features import SKLPolynomialFeatures
    from implementations.feature_preprocessors.random_trees_embedding import SKLRandomTreesEmbedding
    from implementations.feature_preprocessors.rbf_sampler import SKLRBFSampler
    from implementations.feature_preprocessors.select_from_extra_trees_clf import SKLSelectFromExtraTrees
    from implementations.feature_preprocessors.select_from_svc_clf import SKLSelectFromLinearSVC
    from implementations.feature_preprocessors.select_percentile import SKLSelectPercentile
    from implementations.feature_preprocessors.select_rates import SKLSelectRates

    # classifier
    from implementations.classifiers.adaboost import SKLAdaBoost
    from implementations.classifiers.decision_tree import SKLDecisionTree
    from implementations.classifiers.random_forest import SKLRandomForest
    from implementations.classifiers.extra_trees import SKLExtraTrees
    from implementations.classifiers.gradient_boosting import SKLGradientBoosting
    from implementations.classifiers.sgd import SKLSGD
    from implementations.classifiers.svc import SKLKernelSVC

    if multiclass_classification is False:
        from implementations.classifiers.multinominal_nb import SKLMultinomialNB
        from implementations.classifiers.linear_svc import SKLLinearSVC
        from implementations.classifiers.lda import SKLLinearDiscriminantAnalysis
        from implementations.classifiers.knn import SKLKNearestNeighbors
        from implementations.classifiers.gaussian_nb import SKLGaussianNaiveBayes
        from implementations.classifiers.bernoulli_nb import SKLBernoulliNB
        from implementations.classifiers.passive_aggressive import SKLPassiveAggressive
        from implementations.classifiers.qda import SKLQuadraticDiscriminantAnalysis

    if include_categorical is True:
        from implementations.encoding.ordinal_encoder import OrdinalEncoding
        from implementations.encoding.one_hot_encoder import OneHotEncoding
        from implementations.category_coalescence.minority_coalescence import MinorityCoalescence

    if include_string is True:
        pass
