def get_files_list(path):
    files = []
    for root, subdirs, images in os.walk(path):
        if images:
            full_path_images = [os.path.join(root, image).replace("\\", "/") for image in images]
            files.extend(full_path_images)
    return files

def dataPreparation():
    facefiles = get_files_list(dirPath + outputDir + bboxfacesdir)
    nonfacefiles = get_files_list(dirPath + outputDir + bboxnonfacesdir)
    
    facedataimagestr = facefiles[:1000] 
    nonfacedataimagestr = nonfacefiles[:1000]
    
    facedataimageste = facefiles[1000:1100] 
    nonfacedataimageste = nonfacefiles[1000:1100]
    
    facedatatr = [cv2.imread(img) for img in facedataimagestr]
    nonfacedatatr = [cv2.imread(img) for img in nonfacedataimagestr]
    
    facedatate = [cv2.imread(img) for img in facedataimageste]
    nonfacedatate = [cv2.imread(img) for img in nonfacedataimageste]

    facedatatr = np.array([cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten() for i in facedatatr])
    nonfacedatatr = np.array([cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten() for i in nonfacedatatr])
    
    facedatate = np.array([cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten() for i in facedatate])
    nonfacedatate = np.array([cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten() for i in nonfacedatate])
    
    facelabelstr = np.array([1]*1000)
    nonfacelabelstr = np.array([0]*1000)
    
    facelabelste = np.array([1]*100)
    nonfacelabelste = np.array([0]*100)
    
    return facedatatr, nonfacedatatr, facelabelstr, tr_non_face_labels, facedatate, nonfacedatate, facelabelste, nonfacelabelste