import os
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

    al_selection_methods = ["density", "igs", "qbc", "rt", "random"]
    dataset_names = ["Forrester", "forrester_imb", "jump_forrester", "jump_forrester_imb", "gaussian", "gaussian_imb", "gaussian_imb_noise", "exponential", "exponential_imb"]

    dataset_to_plot_names = {"Forrester"          : "Forrester", 
                             "forrester_imb"      : "Forrester Imbalanced", 
                             "jump_forrester"     : "Jump Forrester", 
                             "jump_forrester_imb" : "Jump Forrester Imbalanced", 
                             "gaussian"           : "Gaussian", 
                             "gaussian_imb"       : "Gaussian Imbalanced", 
                             "gaussian_imb_noise" : "Gaussian Imbalanced with Noise", 
                             "exponential"        : "Exponential", 
                             "exponential_imb"    : "Exponential Imbalanced"}

    experiment_results = []
    for dataset in dataset_names:
        for method in al_selection_methods:
            experiment_results.append(create_results('./ALresults/Synthetic/' + method + '/' + dataset + '/' + dataset + '_150_' + method))


        t_v, p_v = stat_test(experiment_results[0],experiment_results[1])
        p_v = p_v.item()

        f_size = 16
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

        save_path = os.curdir + '/plots'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(os.path.join(save_path, dataset + '.png'), dpi=300, bbox_inches='tight')

        experiment_results = []
        # plt.show()
