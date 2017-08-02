from features_extraction.meanRgbGaborExtraction import MeanRgbGaborExtraction
from features_extraction.advancedExtraction import AdvancedExtraction

################ Constructor #################

# Instance a specific features extraction object
def get(type):
    featureType = {'MRG': meanRgbGabor,
                   'ADV': advancedExtraction
                   }
    return featureType[type]()

# Instance a meanRgbGabor object
def meanRgbGabor():
    return MeanRgbGaborExtraction()

def advancedExtraction():
    return AdvancedExtraction()
###########
# Here we can add more functions to instace any feature extraction class
##########

