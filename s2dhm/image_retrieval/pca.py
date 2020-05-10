import numpy as np
from sklearn.decomposition import PCA


def normalize(array, axis=-1):
    """Normalize array."""
    return np.array(array) / np.linalg.norm(array, axis=axis, keepdims=True)

def learn_and_apply_pca(ref_desc, qry_desc, ndim=1024):
    """Learn and apply PCA."""
    pca = PCA(n_components=min(ndim, len(ref_desc)))
    # Learn PCA on reference descriptors
    ref_desc = normalize(pca.fit_transform(normalize(ref_desc)))
    # Apply it on the query descriptors
    qry_desc = normalize(pca.transform(normalize(qry_desc)))
    return ref_desc, qry_desc

def pca_vis(ref_desc, qry_desc, height, width, ndim=3):
    """Learn and apply PCA."""
    # pca = PCA(n_components=min(ndim, len(ref_desc)))
    pca = PCA(n_components=ndim)
    # Learn PCA on reference descriptors
    # ref_rgb = torch.zeros([3,height*width])
    # qry_rgb = torch.zeros([3,height*width])
    # for i in range(ref_desc):
    #     ref_i = pca.fit_transform(ref_desc[:,i])
    #     ref_rgb[]
    # Apply it on the query descriptors
    ref_desc = pca.fit_transform(ref_desc.transpose(0,1))
    qry_desc = pca.fit_transform(qry_desc.transpose(0,1))
    ref_desc = ref_desc.reshape(3,height,width)*255
    qry_desc = qry_desc.reshape(3,height,width)*255

    return ref_desc, qry_desc