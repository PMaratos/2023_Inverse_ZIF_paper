import numpy as np
import matplotlib.pyplot as plt

def plot_logD_trainSize_perMethod(frame1, frame2 = None, frame3 = None, method1_v_method2_stats = None, method1_v_method3_stats = None, label1 = '', label2 = '', label3 = '', on_off = 'False', x_min=0, x_max=75, y_min=0, y_max=10, 
               size='16', line=1.0, edge=2, axes_width = 2, tickWidth = 2, tickLength=12, 
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

    # First Method
    x1 = frame1['sizeOfTrainingSet']
    y1 = frame1['averageError']
    error1 = frame1['stdErrorOfMeanError']

    plt.errorbar(x1, y1, yerr=error1, label=label1, ecolor='k', fmt='o', c=marker_colors[0], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
    plt.legend(loc='upper right', fontsize=15, frameon=on_off)

    # Second Method
    if frame2 is not None:
        x2 = [x + 0.3 for x in frame2['sizeOfTrainingSet']]
        y2 = frame2['averageError']
        error2 = frame2['stdErrorOfMeanError']

        method1_v_method2_text = "\n".join((" ".join(("P-Value:     ", "{:.3e}".format(method1_v_method2_stats["pvalue"]))),
                                            " ".join(("Stat score:"  , "{:.3e}".format(method1_v_method2_stats["statistic"])))))

        plt.errorbar(x2, y2, yerr=error2, label=label2, ecolor='k', fmt='o', c=marker_colors[1], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.errorbar([ ], [ ], None, label=method1_v_method2_text, linestyle='None')
        plt.legend(loc='upper right', fontsize=15, frameon=on_off)

    # Third Method
    if frame3 is not None:
        x3 = [x + 0.4 for x in frame3['sizeOfTrainingSet']]
        y3 = frame3['averageError']
        error3 = frame3['stdErrorOfMeanError']

        method1_v_method3_text = "\n".join((" ".join(("P-Value:     ",    "{:.3e}".format(method1_v_method3_stats["pvalue"]))),
                                            " ".join(("Stat score:", "{:.3e}".format(method1_v_method3_stats["statistic"])))))

        plt.errorbar(x3, y3, yerr=error3, label=label3, ecolor='k', fmt='o', c=marker_colors[2], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.errorbar([ ], [ ], None, label=method1_v_method3_text, linestyle='None')
        plt.legend(loc='upper right', fontsize=15, frameon=on_off)
    

    plt.tick_params(which='both', width=tickWidth)
    plt.tick_params(which='major', length=tickLength)
    # plt.yticks(np.arange(min(y1), max(y1), 0.05 * round(max(y1) * 0.2 / 0.05)))
    plt.xticks(np.arange(0, max(x1) + 5, 5.0))

    plt.savefig(fileName, bbox_inches='tight')
    plt.show()