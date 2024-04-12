The effect of "unlikely wins" can also be anylized quantitatively using the Bradley-Terry methodology. For example, let's consider a tournament where the top two teams have a strength of 9, the middle six teams have an average strength of 1, and the bottom two teams have a strength of 1/9. In such a scenario, the probability of an "unlikely win" (one of the bottom two teams winning against one of the top two teams) is 1.2195\%.

To observe the impact of unlikely wins on the turning point, simulations can be conducted where the "unlikely win" probability is manually set higher than this value. In the following experiment, we averaged the results of 25 simulations across 10 different quadruple-round-robin schedules. The first column denotes the probability we set for an unlikely win, while the second column represents the normalized turning point. As anticipated, tournaments tend to exhibit greater balance, i.e., taking longer to distinguish from purely random ones, as the probability of unlikely wins increases.

| Probability of unlikely win | Average $\tau_\%$ |
|  :-------------------------: |   :---------------:  |
|            1.2195\%        |       16.22\%     |
|              5\%               |       17.60\%     |
|             7.5\%             |       17.63\%     |
|             10\%              |       18.76\%     |
|            12.5\%            |       19.12\%     |

The code for this is in `src/paper_review_results.ipynb`.