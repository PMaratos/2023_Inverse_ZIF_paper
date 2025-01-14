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


    # df1_1 = create_results('./ALresults/Synthetic/density/Forrester/Forrester_150_density')
    # df2_1 = create_results('./ALresults/Synthetic/igs/Forrester/Forrester_150_igs')
    # df3_1 = create_results('./ALresults/Synthetic/qbc/Forrester/Forrester_150_qbc')

    # df1_1 = create_results('./ALresults/Synthetic/density/forrester_imb/forrester_imb_150_density')
    # df2_1 = create_results('./ALresults/Synthetic/igs/forrester_imb/forrester_imb_150_igs')
    # df3_1 = create_results('./ALresults/Synthetic/qbc/forrester_imb/forrester_imb_150_qbc')

    # df1_1 = create_results('./ALresults/Synthetic/density/gaussian/gaussian_150_density')
    # df2_1 = create_results('./ALresults/Synthetic/igs/gaussian/gaussian_150_igs')
    # df3_1 = create_results('./ALresults/Synthetic/qbc/gaussian/gaussian_150_qbc')

    # df1_1 = create_results('./ALresults/Synthetic/density/gaussian_imb/gaussian_imb_150_density')
    # df2_1 = create_results('./ALresults/Synthetic/igs/gaussian_imb/gaussian_imb_150_igs')
    # df3_1 = create_results('./ALresults/Synthetic/qbc/gaussian_imb/gaussian_imb_150_qbc')

    # df1_1 = create_results('./ALresults/Synthetic/density/gaussian_imb_noise/gaussian_imb_noise_150_density')
    # df2_1 = create_results('./ALresults/Synthetic/igs/gaussian_imb_noise/gaussian_imb_noise_150_igs')
    # df3_1 = create_results('./ALresults/Synthetic/qbc/gaussian_imb_noise/gaussian_imb_noise_150_qbc')

    df1_1 = create_results('./ALresults/Synthetic/density/jump_forrester/jump_forrester_150_density')
    df2_1 = create_results('./ALresults/Synthetic/igs/jump_forrester/jump_forrester_150_igs')
    df3_1 = create_results('./ALresults/Synthetic/qbc/jump_forrester/jump_forrester_150_qbc')

    # df1_1 = create_results('./ALresults/Synthetic/density/jump_forrester_imb/jump_forrester_imb_150_density')
    # df2_1 = create_results('./ALresults/Synthetic/igs/jump_forrester_imb/jump_forrester_imb_150_igs')
    # df3_1 = create_results('./ALresults/Synthetic/qbc/jump_forrester_imb/jump_forrester_imb_150_qbc')

    # df1_1 = create_results('./ALresults/Synthetic/density/exponential/exponential_150_density')
    # df2_1 = create_results('./ALresults/Synthetic/igs/exponential/exponential_150_igs')
    # df3_1 = create_results('./ALresults/Synthetic/qbc/exponential/exponential_150_qbc')

    # df1_1 = create_results('./ALresults/Synthetic/density/exponential_imb/exponential_imb_150_density')
    # df2_1 = create_results('./ALresults/Synthetic/igs/exponential_imb/exponential_imb_150_igs')
    # df3_1 = create_results('./ALresults/Synthetic/qbc/exponential_imb/exponential_imb_150_qbc')

    t_v, p_v = stat_test(df1_1,df2_1)
    p_v = p_v.item()

    plt.figure()
    plt.plot(df1_1, label='Density', linewidth=3)
    plt.plot(df2_1, label='iGS', linestyle='--', linewidth=3)
    plt.plot(df3_1, label='QBC', linestyle='-.', linewidth=3)
    #plt.plot(df4_1, label='ActiveLearning RT')
    # plt.plot(df5_1, label='ActiveLearning density')

    plt.xlabel('# of Queries')
    plt.ylabel('MAE')
    plt.title('Average Error Comparison (Forrester)')

    handles, labels = plt.gca().get_legend_handles_labels()

    #labels.append(f'p-value: {p_v:.3e}')
    # labels[0] += f' \n(p-value: {p_v:.3e})'
    plt.legend(handles, labels)
    plt.show()