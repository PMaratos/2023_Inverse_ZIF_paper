import os
import dialogs
import shutil
import statistics
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

def parse_data(path: str, saveName: str):
    total_runs = 0
    stopDataSizeFreq = {}
    stopDataSizeFreqThres = {}
    stopDataSizeFreqPerf = {}
    lowPerformanceZifs = {}
    thresholdReachingZifs = {}
    for experimentFile in os.listdir(path):

        # Skip non-directory files
        if not os.path.isdir(os.path.join(path, experimentFile)):
            continue

        savePath = os.path.join(path, experimentFile, saveName)
        if not os.path.isdir(savePath):
            continue

        for roundDir in os.listdir(savePath):

            # Skip non-directory files
            if not os.path.isdir(os.path.join(savePath, roundDir)):
                continue

            total_runs += 1

            selectedDataset = None
            for roundResult in os.listdir(os.path.join(savePath, roundDir)):

                fileSplit = roundResult.split('_')
                # Skip the full round report.
                if fileSplit[0] != 'full':

                    if (selectedDataset is None) or (fileSplit[-2] > selectedDataset.split('_')[-2]):
                        selectedDataset = os.path.join(savePath, roundDir, roundResult)

            stoppedDataSize = selectedDataset.split('_')[-2]
            if stoppedDataSize in stopDataSizeFreq:
                stopDataSizeFreq[stoppedDataSize] += 1
            else:
                stopDataSizeFreq[stoppedDataSize] = 1

            data = pd.read_csv(selectedDataset)            
            tested_zif = data["tested_against"][0]
            if data["stop_criterion"][1] == "error_threshold_reached":
                if stoppedDataSize in stopDataSizeFreqThres:
                    stopDataSizeFreqThres[stoppedDataSize] += 1
                else:
                    stopDataSizeFreqThres[stoppedDataSize] = 1
                if tested_zif in thresholdReachingZifs:
                    thresholdReachingZifs[tested_zif]["count"] += 1
                    thresholdReachingZifs[tested_zif]["datasets"].append(list(data["type"].unique()))
                else:
                    thresholdReachingZifs[tested_zif] = {"count": 1,
                                                         "datasets": [list(data["type"].unique())]}
            else:
                if stoppedDataSize in stopDataSizeFreqPerf:
                    stopDataSizeFreqPerf[stoppedDataSize] += 1
                else:
                    stopDataSizeFreqPerf[stoppedDataSize] = 1
                
                if tested_zif in lowPerformanceZifs:
                    lowPerformanceZifs[tested_zif]["count"] += 1
                    lowPerformanceZifs[tested_zif]["datasets"].append(list(data["type"].unique()))
                else:
                    lowPerformanceZifs[tested_zif] = {"count": 1,
                                                      "datasets": [list(data["type"].unique())]}
                    

    sortedDataSizeFreqPerf  = {k: stopDataSizeFreqPerf[k] for k in sorted(stopDataSizeFreqPerf, key=lambda x: float(x))}
    sortedDataSizeFreq = {k: stopDataSizeFreq[k] for k in sorted(stopDataSizeFreq, key=lambda x: float(x))}
    mostFreqDataSize = list(sortedDataSizeFreq.keys())[0]

    number_of_stoped_runs = 0
    for key in thresholdReachingZifs.keys():
        number_of_stoped_runs += thresholdReachingZifs[key]["count"]
    for key in lowPerformanceZifs.keys():
        number_of_stoped_runs += lowPerformanceZifs[key]["count"]

    return sortedDataSizeFreq, stopDataSizeFreqThres, stopDataSizeFreqPerf, mostFreqDataSize, thresholdReachingZifs, lowPerformanceZifs, total_runs

def plot_mae_per_size(data: dict):
    plt.bar(list(data.keys()), list(data.values()), width = 0.6)
    plt.xlabel('Number of Zifs in Dataset')
    plt.ylabel('Number of times stop criteria were met.')
    plt.show()

def box_plot_statistics(mae_data: list, case: str):
    print("\n" + "Statistics for " + case + "\n")
    print("Minimum:", min(mae_data))
    print("Maximum:", max(mae_data))
    print("Median:",  statistics.median(mae_data))
    print("Mean:",    statistics.mean(mae_data))
    print("25th Percentile:", statistics.quantiles(mae_data, n=4)[0])
    print("75th Percentile:", statistics.quantiles(mae_data, n=4)[2])

    plt.boxplot(mae_data)
    plt.yticks(np.arange(0, max(mae_data), 0.5))
    plt.ylabel('Mean Absolute Error')
    plt.title('Box Plot of '+ case)
    plt.show()

def cumulative_thres(threshold_criterion_results: dict, numberOfRuns: int, plot: bool = True):

    sortedData = {k: threshold_criterion_results[k] for k in sorted(threshold_criterion_results, key=lambda x: float(x))}

    prevSum = 0
    for key in sortedData.keys():
        sortedData[key] += prevSum
        prevSum = sortedData[key]
        sortedData[key] /= numberOfRuns

    if plot:
        plt.bar(sortedData.keys(), sortedData.values())
        plt.show()
    else:
        return sortedData

def succes_rate_per_zif(success: dict, failure: dict):

    init()

    sorted_success = dict(sorted(success.items(), key=lambda x: x[1]["count"], reverse=True))

    tmp_failure = failure.copy()
    for key in sorted_success.keys():
        if key in tmp_failure.keys():
            print("ZIF: " + key + Fore.GREEN + " Success Rate: " + Style.RESET_ALL + str(sorted_success[key]["count"] / (sorted_success[key]["count"] + tmp_failure[key]["count"])) + "(" + str(sorted_success[key]["count"] + tmp_failure[key]["count"]) + ")")
            del tmp_failure[key]
        else:
            print("ZIF: " + key + Fore.GREEN + ", Succeded " + Style.RESET_ALL + str(sorted_success[key]["count"]) + " times and never failed.")

    for key in tmp_failure.keys():
        print("ZIF: " + key + Fore.RED + " Failed " + Style.RESET_ALL + str(tmp_failure[key]["count"]) + " times.")

    print("Total number of ZIFs that achieved small error: " + str(len(success)))
    print("Total number of ZIFs that did not achieve small  error: " + str(len(tmp_failure)))

def datasets_used_against_zif(success: dict, failure: dict):

    print("Enter zif name.")
    testZif = input()

    init()

    if testZif not in success.keys() and testZif not in failure.keys():
        print("The ZIF name you entered does not exist.")
    else:
        if testZif in success.keys():
            print(Fore.GREEN + "Succeded" + Style.RESET_ALL)
            for x in success[testZif]["datasets"]:
                print(x)

        if testZif in failure.keys():
            print(Fore.RED + "Failed" + Style.RESET_ALL)
            for x in failure[testZif]["datasets"]:
                print(x)    

def get_datasets_by_probability(threshold_criterion_results: dict, total_runs: int, path: str, saveName: str):
    
    single_prob = False
    print("Do you want to get datasets coresponding to one probability? [Y/N] ")
    if dialogs.yesNoInput():
        single_prob = True
        print("Please enter the probability to search for.")
    else:
        print("Please enter the lower bound of the probability to search for.")
    probability = float(input())

    cumulative_results = cumulative_thres(threshold_criterion_results, total_runs, plot = False)
    key_prob_distance = {}
    if single_prob:
        for key in list(cumulative_results.keys()):
            key_prob_distance[key] = abs(cumulative_results[key] - probability)
        sorted_key_prob_distance = dict(sorted(key_prob_distance.items(), key=lambda x: x[1], reverse=True))

        key_list = list(sorted_key_prob_distance.keys())[:-1]
        for key in key_list:
            del cumulative_results[key]
    else:
        for key in list(cumulative_results.keys()):
            if cumulative_results[key] < probability:
                del cumulative_results[key]

    if len(cumulative_results) == 0:
        print("There are no datasets that meet the given probability threshold.")
        return

    selected_datasets_path = os.path.join(os.curdir, "selected_datasets")
    if os.path.exists(selected_datasets_path):
        shutil.rmtree(os.path.join(selected_datasets_path,"",""))

    os.mkdir(selected_datasets_path)

    overThresCount = 0
    totalCount = 0
    for experimentFile in os.listdir(path):

        # Skip non-directory files
        if not os.path.isdir(os.path.join(path, experimentFile)):
            continue

        # Skip non-directory files
        savePath = os.path.join(path, experimentFile, saveName)
        if not os.path.isdir(savePath):
            continue

        for roundDir in os.listdir(savePath):

            # Skip non-directory files
            if not os.path.isdir(os.path.join(savePath, roundDir)):
                continue                

            selectedDataset = None
            for roundResult in os.listdir(os.path.join(savePath, roundDir)):

                fileSplit = roundResult.split('_')
                # Skip the full round report.
                if fileSplit[0] != 'full':

                    if (selectedDataset is None) or (fileSplit[-2] > selectedDataset.split('_')[-2]):
                        selectedDataset = os.path.join(savePath, roundDir, roundResult)

            if selectedDataset is None:
                continue    
            dataSize = selectedDataset.split('_')[-2]
            if dataSize not in list(cumulative_results.keys()):
                continue

            data = pd.read_csv(selectedDataset)
            if data["mae"][0] > 0.5:
                overThresCount += 1
            
            totalCount += 1

            data_size_dir = os.path.join(selected_datasets_path, dataSize)
            if not os.path.exists(data_size_dir):
                os.mkdir(data_size_dir)
            
            shutil.copy(selectedDataset, data_size_dir)
    
    print("Numerator of the percentage: " + str(overThresCount))
    print("Denominator of the percentage: " + str(totalCount))
    print("The percentage of datasets that achieved large error is: " + str(overThresCount / totalCount))

def analyse_by_data_size(most_freq_size: int, path: str, saveName: str):

        print("Which data size results do you want to analyse?")
        selected_size = input()

        if selected_size == "-1":
            selected_size = most_freq_size

        value_range = {}
        for experimentFile in os.listdir(path):

            # Skip non-directory files
            if not os.path.isdir(os.path.join(path, experimentFile)):
                continue

            # Skip non-directory files
            savePath = os.path.join(path, experimentFile, saveName)
            if not os.path.isdir(savePath):
                continue

            for roundDir in os.listdir(savePath):

                # Skip non-directory files
                if not os.path.isdir(os.path.join(savePath, roundDir)):
                    continue                

                selectedDataset = None
                for roundResult in os.listdir(os.path.join(savePath, roundDir)):

                    fileSplit = roundResult.split('_')
                    # Skip the full round report.
                    if fileSplit[0] != 'full':

                        if (selectedDataset is None) or (fileSplit[-2] > selectedDataset.split('_')[-2]):
                            selectedDataset = os.path.join(savePath, roundDir, roundResult)

                if selectedDataset is None:
                    continue    
                dataSize = selectedDataset.split('_')[-2]
                if dataSize != selected_size:
                    continue

                data = pd.read_csv(selectedDataset)
                value_range[data["mae"][1]] = {"path": selectedDataset,
                                                        "stop_criterion": data["stop_criterion"][1],
                                                        "tested_zif": data["tested_against"][1]}

        # Plot number of times each stop criterion was activated.
        thresholdCount   = 0
        performanceCount = 0
        thresholdMae   = []
        performanceMae = []
        for mae, info in value_range.items():
            if info["stop_criterion"] == "error_threshold_reached":
                thresholdCount += 1
                thresholdMae.append(mae)
            if info["stop_criterion"] == "low_performance_gain":
                performanceCount += 1
                performanceMae.append(mae)

        plt.bar(["threshold", "performance"], [thresholdCount, performanceCount], width = 0.1)
        plt.xlabel('Number of Zifs in Dataset')
        plt.ylabel('Number of times any stop criterion was met.')
        plt.show()

        if len(value_range) > 1:
            if thresholdMae:
                box_plot_statistics(thresholdMae, "Threshold Criterion")
            if performanceMae:
                box_plot_statistics(performanceMae, "No Performance Gain Criterion")
            if value_range:
                box_plot_statistics(value_range.keys(), "Range of Values")
        else:
            print("Can not further analyse data. Multitude less than 2.")