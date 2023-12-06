from .classifiers.knn import SKLKNearestNeighbors
from .classifiers.gaussian_nb import SKLGaussianNaiveBayes
from .classifiers.multinominal_nb import SKLMultinomialNB
from .classifiers.svc import SKLKernelSVC
from .classifiers.decision_tree import SKLDecisionTree
from .classifiers.adaboost import SKLAdaBoost
from .classifiers.gradient_boosting import SKLGradientBoosting
from .classifiers.extra_trees import SKLExtraTrees
from .classifiers.random_forest import SKLRandomForest
from .classifiers.qda import SKLQuadraticDiscriminantAnalysis
from .classifiers.lda import SKLLinearDiscriminantAnalysis

from .feature_preprocessors.fast_ica import SKLFastICA
from .feature_preprocessors.kernel_pca import SKLKernelPCA
from .feature_preprocessors.nystroem import SKLNystroem
from .feature_preprocessors.pca import SKLPCA
from .feature_preprocessors.polynomial_features import SKLPolynomialFeatures
from .feature_preprocessors.random_trees_embedding import SKLRandomTreesEmbedding
from .feature_preprocessors.rbf_sampler import SKLRBFSampler
from .feature_preprocessors.select_percentile import SKLSelectPercentile
from .feature_preprocessors.select_rates import SKLSelectRates
from .feature_preprocessors.select_from_extra_trees_clf import SKLSelectFromExtraTrees
from .feature_preprocessors.select_from_svc_clf import SKLSelectFromLinearSVC
from .feature_preprocessors.feature_agglomeration import SKLFeatureAgglomeration
from .feature_preprocessors.no_feature_preprocessor import NoFeaturePreprocessor



FORBIDDEN = [
    [SKLMultinomialNB, SKLNystroem],
    [SKLMultinomialNB, SKLKernelPCA],
    [SKLMultinomialNB, SKLFastICA],
    [SKLMultinomialNB, SKLPCA],
    [SKLMultinomialNB, SKLRBFSampler],
    [SKLGaussianNaiveBayes, SKLNystroem],
    [SKLGaussianNaiveBayes, SKLRBFSampler],
    [SKLGaussianNaiveBayes, SKLKernelPCA],
    [SKLRandomForest, SKLNystroem],
    [SKLRandomForest, SKLRBFSampler],
    [SKLRandomForest, SKLKernelPCA],
    [SKLKernelSVC, SKLNystroem],
    [SKLKernelSVC, SKLRBFSampler],
    [SKLKernelSVC, SKLKernelPCA],
    [SKLKNearestNeighbors, SKLNystroem],
    [SKLKNearestNeighbors, SKLRBFSampler],
    [SKLKNearestNeighbors, SKLKernelPCA],
    [SKLGradientBoosting, SKLNystroem],
    [SKLGradientBoosting, SKLRBFSampler],
    [SKLGradientBoosting, SKLKernelPCA],
    [SKLExtraTrees, SKLNystroem],
    [SKLExtraTrees, SKLRBFSampler],
    [SKLExtraTrees, SKLKernelPCA],
    [SKLDecisionTree, SKLNystroem],
    [SKLDecisionTree, SKLRBFSampler],
    [SKLDecisionTree, SKLKernelPCA],
    [SKLAdaBoost, SKLNystroem],
    [SKLAdaBoost, SKLRBFSampler],
    [SKLAdaBoost, SKLKernelPCA],
    [SKLMultinomialNB, SKLSelectRates],
    [SKLMultinomialNB, SKLSelectPercentile],
    [SKLQuadraticDiscriminantAnalysis, SKLRandomTreesEmbedding],
    [SKLLinearDiscriminantAnalysis, SKLRandomTreesEmbedding],
    [SKLGradientBoosting, SKLRandomTreesEmbedding],
    [SKLGaussianNaiveBayes, SKLRandomTreesEmbedding],
    [SKLMultinomialNB, SKLPolynomialFeatures],
    [SKLMultinomialNB, NoFeaturePreprocessor],
    [SKLMultinomialNB, SKLSelectFromLinearSVC],
    [SKLMultinomialNB, SKLFeatureAgglomeration],
    [SKLMultinomialNB, SKLSelectFromExtraTrees],
]
