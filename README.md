For this project, the E. coli dataset of the Institue of Molecular and Cellular Biology of Osaka, University) has been used. The dataset had 106 DNA sequences, with 57 sequential nucleotides each. The aim was to predict whether a short sequence in the DNA of the E. coli bacteria was a promoter or not.

A promoter is a region of DNA where the transcription of a gene is initiated. Promoters are a component of expression vectors because they control the binding of RNA polymerase to DNA. RNA polymerase transcribes DNA to mRNA which is ultimately translated into a functional protein. Thus the promoter region controls when and where in the organism your gene of interest is expressed.

Different classification algorithms were for this binary task applied and trained, among others: K-NN, Decision Tree, Random Forest and Naive Bayes. Eventually the SVM Linear came out to be the best predictor with the highest precision (97.00%) and f1 score (96.50%).
