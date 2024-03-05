import random 
import torch
from torch.distributions.multinomial import Multinomial
import matplotlib.pyplot as plt

#2.6.1 Tossing Coins
number_of_coin_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(number_of_coin_tosses)]) # Gets amount of times random number is greater than 0.5
tails = number_of_coin_tosses-heads
print("heads, tails: ", [heads, tails])
# The probability here shoudl be close to 50-50. It will not always be the case however.


probabilities = torch.tensor([0.01, 0.01, 0.5, 1]) # tensor of probabilities 
multinominal = Multinomial(100, probabilities).sample() 
""" 
returns torch tensor with each index correlating to the index of 
the probabilities and how many times the given index was picked out of the sample size.
tricky beause probability of one(seen above) will not be [0, 0, 0, 100]
it is roughly torch.tensor([3, 2, 31, 64]) so each probability relates to eachother
for example, with probabilities [0.1, 0.1] it will give 
a result of roughly [50, 50], so even if probabilities were 0.1, they were all equal thus fair probabilities
"""
print(multinominal)


probs = Multinomial(1, torch.tensor([0.5, 0.5])).sample((10000, ))
cum_sum = probs.cumsum(dim=0) 
cum_sum = cum_sum / cum_sum.sum(dim=1, keepdim=True)
print(cum_sum)
plt.plot([x for x in range(10000)], cum_sum[:, 0])
plt.plot([x for x in range(10000)], cum_sum[:, 1])
plt.xlabel('Samples')
plt.ylabel('Estimated Probability')
plt.show()
"""
The code above is about the law of large numbers
and the central limit theorem that the error in samples and their predicted probability
should decrease at a rate of 1/sqrt(n) where n is amount of samples

Uses Matplotlib instead of d2l
"""


# 2.6.2 A More Formal Treatment

# no code here just math and reading

# NEED HELP UNDERSTANDING...

# Random Variables

# no code just reading and math


# MATH IS CONFUSING...









