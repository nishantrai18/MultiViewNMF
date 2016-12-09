Partial Multi View Clustering using Non Negative Matrix Factorization:
======================================================================

**Authors:**

- Nishant Rai, Indian Institute of Technology Kanpur
- Sumit Negi, Amazon Development Center
- Santanu Chaudhury, Indian Institute of Technology Delhi 
- Om Deshmukh, Xerox Research Centre India

**Accepted in ICPR '16 (International Conference on Pattern Recognition)**

The project deals with the problem of clustering data using information present in Multiple Views. The repository contains the relevant code to recreate the results in the paper. The abstract is provided below,

> *Abstract* — Real-world datasets consist of data representations (views) from different sources which often provide information complementary to each other. Multi-view learning algorithms aim at exploiting the complementary information present in different views for clustering and classification tasks. Several multi-view clustering methods that aim at partitioning objects into clusters based on multiple representations of the object have been proposed. Almost all of the proposed methods assume that each example appears in all views or at least there is one view containing all examples. In real-world settings this assumption might be too restrictive. Recent work on Partial View Clustering addresses this limitation by proposing a Non negative Matrix Factorization based approach called PVC. Our work extends the PVC work in two directions. First, the current PVC algorithm is designed specifically for two-view datasets. We extend this algorithm for the k partial-view scenario. Second, we extend our k partial-view algorithm to include view specific graph laplacian regularization. This enables the proposed algorithm to exploit the intrinsic geometry of the data distribution in each view. The proposed method, which is referred to as GPMVC (Graph Regularized Partial Multi-View Clustering), is compared against 7 baseline methods (including PVC) on 5 publicly available text and image datasets. In all settings the proposed GPMVC method outperforms all baselines. For the purpose of reproducibility, we provide access to our code.

*Reproducing Results*: Refer to scripts in gMVNMF/GPVC_B/. Details provided in the readme
