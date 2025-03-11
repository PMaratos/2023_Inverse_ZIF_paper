import os
import argparse
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Helper method for creating average MAE across 10 experiments
def create_results(res_path):

    file_path = res_path + '_0.csv'
    if os.path.exists(file_path) == False:
        print("File: " + file_path + " does not exist.")
        exit()


    df = pd.read_csv(res_path + '_0.csv')
    df = df[['averageError']]
    res = df
    for i in range(1,10):
        file_path = res_path + f'_{i}.csv'
        if os.path.exists(file_path) == False:
            print("File: " + file_path + " does not exist.")
            exit()

        df = pd.read_csv(file_path)
        df = df[['averageError']]
        res += df

    return res / 10    

# Helper method for calculating p-value statistic
def stat_test(df1,df2):
    
    arr1 = df1.to_numpy()
    arr2 = df2.to_numpy()

    t_stat, p_value = stats.ttest_rel(arr1, arr2)
    return t_stat,p_value


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--type', help="The type of data to plot, either real or synthetic.", default='synthetic')
    parser.add_argument("-d", '--data', help="The the path to the dataset.",                        default="./ALresults/Synthetic/")

    args = parser.parse_args()

    data_type = args.type
    data_path = args.data

    al_selection_methods = ["density", "igs", "qbc", "rt", "random"]

    dataset_names = None
    dataset_to_plot_names = None
    if data_type == "real":
        dataset_names = ["CH4", "H2", "He", "N2", "O2"]

        dataset_to_plot_names = {}
        for name in dataset_names:
            dataset_to_plot_names[name] = name

        data_path = './ALresults/Real/'

    else:
        dataset_names = ["Forrester", "forrester_imb", "jump_forrester", "jump_forrester_imb", "gaussian", "gaussian_imb", "gaussian_imb_noise", "exponential", "exponential_imb"]

        dataset_to_plot_names = {"Forrester"          : "Forrester", 
                                 "forrester_imb"      : "Forrester Heterogeneous", 
                                 "jump_forrester"     : "Jump Forrester", 
                                 "jump_forrester_imb" : "Jump Forrester Heterogeneous", 
                                 "gaussian"           : "Gaussian", 
                                 "gaussian_imb"       : "Gaussian Heterogeneous", 
                                 "gaussian_imb_noise" : "Gaussian Heterogeneous with Noise", 
                                 "exponential"        : "Exponential", 
                                 "exponential_imb"    : "Exponential Heterogeneous"}

    experiment_results = []
    for dataset in dataset_names:
        for method in al_selection_methods:
            experiment_results.append(create_results(data_path + method + '/' + dataset + '/' + dataset + '_150_' + method))


        t_v, p_v = stat_test(experiment_results[0],experiment_results[1])
        p_v = p_v.item()

        f_size = 13
        linewidth = 2.2

        plt.figure()
        plt.plot(experiment_results[0], label='DAGS', linewidth=linewidth)
        plt.plot(experiment_results[1], label='iGS', linestyle='--', linewidth=linewidth)
        plt.plot(experiment_results[2], label='QBC', linestyle='-.', linewidth=linewidth)
        plt.plot(experiment_results[3], label='RT',  linestyle=':', linewidth=linewidth)
        plt.plot(experiment_results[4], label='Random', linestyle=(0, (3, 2)), linewidth=linewidth)

        plt.xlabel('# of Queries', fontsize=f_size)
        plt.ylabel('MAE', fontsize=f_size)
        plt.title('Average Error Comparison (' + dataset_to_plot_names[dataset] + ')', fontsize=f_size)

        plt.xticks(fontsize=f_size)
        plt.xticks(fontsize=f_size)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, fontsize=f_size)

        save_path = os.path.join(os.curdir, 'plots', data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(os.path.join(save_path, dataset + '.png'), dpi=300, bbox_inches='tight')

        experiment_results = []
        # plt.show()
