from features_extraction.meanRgbGaborExtraction import MeanRgbGaborExtraction

################ Constructor #################

# Instance a specific features extraction object
def get(type):
    featureType = {'MRG': meanRgbGabor
                   }
    return featureType[type]()

# Instance a meanRgbGabor object
def meanRgbGabor():
    return MeanRgbGaborExtraction()

###########
# Here we can add more functions to instace any feature extraction class
##########

