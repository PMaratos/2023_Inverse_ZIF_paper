import numpy as np
import matplotlib.pyplot as plt

def plot_logD_trainSize_perMethod(frame1, frame2 = None, frame3 = None, bo_v_random_stats = None, bo_v_serial_stats = None, label1 = '', label2 = '', label3 = '', on_off = 'False', x_min=0, x_max=75, y_min=0, y_max=10, 
               size='16', line=2.5, edge=2, axes_width = 2, tickWidth = 2, tickLength=12, 
            xLabel = '', yLabel ='', fileName = 'picture.png', marker_colors = ['y', 'g', 'r']):
    
    """ Plot the Mean (MAE) of logD (y-axis) to Size of Training Dataset (x-axis) for up to three methods 
        frame1 - 3:     A dataframe containing the follwing Columns:
                                                                    1.  sizeOfTrainingSet
                                                                    2.  averageError
                                                                    3.  stdErrorOfMeanError
        label1 - 3:     The name of the respective method used.
        on_off:         The value of frameon argument for pyplot.legend function.
        x_min:          The minimum value of x-axis
        x_max:          The maximum value of x-axis
        y_min:          The minimum value of y-axis
        y_max:          The maximum value of y-axis
        xLabel:         The label of x-axis
        yLabel:         The label of y-axis
        fileName:       The name under which the plot will be saved.
        marker_colors:  The colors that distinguish each method.
        """

    sub_plot_Num = '1'
    if frame3 is not None:
        sub_plot_Num = '2'

    fig, ax = plt.subplots(nrows=int(sub_plot_Num), ncols=1, sharex=True, sharey=True)

    fig.text(0.5, 0.04, xLabel, ha='center', fontsize=16)
    fig.text(0.04, 0.5, yLabel, va='center', rotation='vertical', fontsize=16)

    # First Method
    x1 = frame1['sizeOfTrainingSet']
    y1 = frame1['averageError']
    error1 = frame1['stdErrorOfMeanError']

    plt.subplot(int(sub_plot_Num + "11"))
    plt.errorbar(x1, y1, yerr=error1, label=label1, ecolor='k', fmt='o', c=marker_colors[0], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
    plt.legend(loc='upper right', fontsize=15, frameon=on_off)

    # Second Method
    if frame2 is not None:
        x2 = [x + 0.35 for x in frame2['sizeOfTrainingSet']]
        y2 = frame2['averageError']
        error2 = frame2['stdErrorOfMeanError']
        # plt.subplot(int(sub_plot_Num + "12"))
        plt.errorbar(x2, y2, yerr=error2, label=label2, ecolor='k', fmt='o', c=marker_colors[1], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.legend(loc='upper right', fontsize=15, frameon=on_off)

        plt.tick_params(which='both', width=tickWidth)
        plt.tick_params(which='major', length=tickLength)
        plt.yticks()
        plt.xticks(np.arange(x_min, x_max, 5.0))

    # Third Method
    if frame3 is not None:
        x3 = [x + 0.4 for x in frame3['sizeOfTrainingSet']]
        y3 = frame3['averageError']
        error3 = frame3['stdErrorOfMeanError']
        
        plt.subplot(int(sub_plot_Num + "12"))
        plt.errorbar(x1, y1, yerr=error1, label=label1, ecolor='k', fmt='o', c=marker_colors[0], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.errorbar(x3, y3, yerr=error3, label=label3, ecolor='k', fmt='o', c=marker_colors[2], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.legend(loc='upper right', fontsize=15, frameon=on_off)
    
    bo_v_random_text = ""
    if bo_v_random_stats is not None:
        bo_v_random_text = "\n".join((bo_v_random_text,
                                    "Bayesian vs Random",
                                    " ".join(("P-Value:",    "{:.3e}".format(bo_v_random_stats["pvalue"]))),
                                    " ".join(("Stat score:", "{:.3e}".format(bo_v_random_stats["statistic"])))))

    bo_v_serial_text = ""
    if bo_v_serial_stats is not None:
        bo_v_serial_text = "\n".join((bo_v_serial_text,
                                    "\n",
                                    "Bayesian vs Serial",
                                    " ".join(("P-Value:",    "{:.3e}".format(bo_v_serial_stats["pvalue"]))),
                                    " ".join(("Stat score:", "{:.3e}".format(bo_v_serial_stats["statistic"])))))

    # if stat_test_text != "":
    #     ax =plt.subplot()
    #     plt.text(0.83, 0.8, stat_test_text,
    #     fontsize = 14,
    #     bbox = dict(facecolor='none', edgecolor='grey', pad=5.0),
    #     horizontalalignment='left',
    #     verticalalignment='top',
    #     transform = ax.transAxes)

    plt.tick_params(which='both', width=tickWidth)
    plt.tick_params(which='major', length=tickLength)
    plt.yticks()
    plt.xticks(np.arange(x_min, x_max, 5.0))

    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    # plt.savefig(fileName, bbox_inches='tight')
    plt.show()