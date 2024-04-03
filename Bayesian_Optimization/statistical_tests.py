from logger import Logger
from scipy import stats

class Statistical_Tests():
    
    def __init__(self,testName,logger: Logger):
        self.logger = logger
        self.logPrefix = "Statistical Tests"
        self.name = testName

    def getTest(self, arrayA, arrayB):
        
        if self.name == "pairedT":
            return self.getPairedT(arrayA, arrayB)
        else:
            raise NotImplementedError("No other Statistical Test Implemented Yet.")

    def getPairedT(self,arrayA, arrayB):
        self.logger.info(self.logPrefix, "Performing Paired T Test")
        results = stats.ttest_rel(arrayA,arrayB)
        self.logger.info(self.logPrefix, "Paired T Test Completed With p-value: " + str(results.pvalue) + " and statistic: " + str(results.statistic))
        return results