import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({'font.size': 26})

lt = np.load("./loss_list_t_loss.npy") / 10000
lnt = np.load("./loss_list_no_t_loss.npy") /10000

loss_diff = np.mean(lnt-lt)
diff_percent = 100 * loss_diff / np.mean(lt)

print("diff_percent",diff_percent)

idxs = np.arange(1,104)
fig = plt.figure(1,figsize=(16,9))

plt.plot(idxs[0:30],lt[0:30],'g-o')
plt.plot(idxs[0:30],lnt[0:30],'r-x')

# plt.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel("Frame Number")
plt.ylabel("Normalised MSE Loss")

plt.legend(['With temporal loss','Without temporal loss'])
plt.title("MSE loss between warped and actual image for the test set")

plt.savefig("loss vs frame.tif")

